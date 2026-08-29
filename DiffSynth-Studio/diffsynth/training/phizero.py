from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import Dataset

from ..diffusion import FlowMatchScheduler
from ..models.phizero import (
    PhiZeroModelConfig,
    PhysicalLanguageDecoderConditioner,
    PhysicalLanguageTokenizer,
    export_phi_zero_state_dict,
    load_phi_zero_lora,
    load_physical_language_decoder,
    load_physical_language_tokenizer,
)
from ..pipelines.phizero import load_wan_base_models, predict_physical_language_flow


class VideoPathDataset(Dataset):
    """Deterministic video loader for a CSV with one column named ``video``."""

    def __init__(self, csv_path: str | Path, num_frames: int, height: int, width: int) -> None:
        self.csv_path = Path(csv_path).resolve()
        with self.csv_path.open(newline='') as file:
            self.video_paths = [
                (self.csv_path.parent / row['video']).resolve()
                for row in csv.DictReader(file)
            ]
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        reader = imageio.get_reader(self.video_paths[index])
        try:
            frames = []
            for frame in reader:
                frames.append(torch.from_numpy(np.asarray(frame)).permute(2, 0, 1))
                if len(frames) == self.num_frames:
                    break
        finally:
            reader.close()
        if len(frames) != self.num_frames:
            raise ValueError(f'{self.video_paths[index]} has fewer than {self.num_frames} frames.')

        video = torch.stack(frames, dim=1).float().mul_(2.0 / 255.0).sub_(1.0)
        if video.shape[-2:] != (self.height, self.width):
            video = rearrange(video, 'c t h w -> t c h w')
            video = F.interpolate(
                video,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False,
            )
            video = rearrange(video, 't c h w -> c t h w')
        return video


class PhiZeroDiffusionTrainingModel(nn.Module):
    """Reconstruction training for decoder refinement or source-domain adaptation."""

    def __init__(
        self,
        dit: nn.Module,
        vae: nn.Module,
        tokenizer: PhysicalLanguageTokenizer,
        conditioner: PhysicalLanguageDecoderConditioner,
        model_config: PhiZeroModelConfig,
        lora_module_names: tuple[str, ...],
        *,
        torch_dtype: torch.dtype,
        motion_height: int,
        motion_width: int,
        motion_dropout: float,
        gradient_checkpointing: bool,
        train_tokenizer: bool,
        fsq_entropy_weight: float,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.vae = vae.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.conditioner = conditioner
        self.model_config = model_config
        self.lora_module_names = lora_module_names
        self.torch_dtype = torch_dtype
        self.motion_height = motion_height
        self.motion_width = motion_width
        self.motion_dropout = motion_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.train_tokenizer = train_tokenizer
        self.fsq_entropy_weight = fsq_entropy_weight
        if not train_tokenizer:
            self.tokenizer.eval().requires_grad_(False)
        self.scheduler = FlowMatchScheduler('Wan')
        self.scheduler.set_timesteps(1000, training=True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        dit_path: str | Sequence[str] | None,
        vae_path: str | None,
        device: str | torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        motion_height: int = 256,
        motion_width: int = 448,
        motion_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        train_tokenizer: bool = False,
        fsq_entropy_weight: float = 0.0,
    ) -> 'PhiZeroDiffusionTrainingModel':
        model_config = PhiZeroModelConfig.from_checkpoint(checkpoint)
        dit, vae = load_wan_base_models(device, torch_dtype, dit_path, vae_path)
        lora_module_names = load_phi_zero_lora(dit, checkpoint)

        with torch.device('meta'):
            tokenizer = PhysicalLanguageTokenizer(model_config.motion_tokenizer)
        conditioner = PhysicalLanguageDecoderConditioner(model_config)
        load_physical_language_tokenizer(tokenizer, checkpoint)
        load_physical_language_decoder(conditioner, checkpoint)
        tokenizer = tokenizer.to(device=device, dtype=torch_dtype)
        conditioner = conditioner.to(device=device, dtype=torch_dtype)
        return cls(
            dit,
            vae,
            tokenizer,
            conditioner,
            model_config,
            lora_module_names,
            torch_dtype=torch_dtype,
            motion_height=motion_height,
            motion_width=motion_width,
            motion_dropout=motion_dropout,
            gradient_checkpointing=gradient_checkpointing,
            train_tokenizer=train_tokenizer,
            fsq_entropy_weight=fsq_entropy_weight,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if not self.train_tokenizer:
            self.tokenizer.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        return export_phi_zero_state_dict(
            self.tokenizer,
            self.conditioner,
            self.dit,
            self.lora_module_names,
        )

    def _tokenizer_input(self, video: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _, _ = video.shape
        motion_video = rearrange(video, 'b c t h w -> (b t) c h w')
        motion_video = F.interpolate(
            motion_video,
            size=(self.motion_height, self.motion_width),
            mode='bilinear',
            align_corners=False,
        )
        motion_video = rearrange(
            motion_video,
            '(b t) c h w -> b c t h w',
            b=batch,
            t=frames,
        )
        return motion_video

    def _physical_language_context(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion_video = self._tokenizer_input(video)
        if self.train_tokenizer:
            if self.fsq_entropy_weight:
                _, codes, entropy_loss = self.tokenizer.encode_for_training(motion_video)
            else:
                _, codes = self.tokenizer.encode(motion_video)
                entropy_loss = video.new_zeros(())
            return self.conditioner.from_codes(codes), entropy_loss
        with torch.no_grad():
            physical_language = self.tokenizer(motion_video)
        return self.conditioner(physical_language), video.new_zeros(())

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = video.to(dtype=self.torch_dtype)
        with torch.no_grad():
            input_latents = self.vae.encode(video, device=video.device).to(self.torch_dtype)
            first_frame_latents = self.vae.encode(video[:, :, :1], device=video.device).to(
                self.torch_dtype
            )

        context, entropy_loss = self._physical_language_context(video)
        drop_mask = torch.rand(context.shape[0], device=context.device) < self.motion_dropout
        context = torch.where(drop_mask[:, None, None], torch.zeros_like(context), context)

        timestep_id = torch.randint(0, len(self.scheduler.timesteps), (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            device=video.device,
            dtype=self.torch_dtype,
        )
        noise = torch.randn_like(input_latents)
        latents = self.scheduler.add_noise(input_latents, noise, timestep)
        latents[:, :, :1] = first_frame_latents

        prediction = predict_physical_language_flow(
            self.dit,
            latents,
            timestep,
            context,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        target = noise - input_latents
        loss = F.mse_loss(prediction[:, :, 1:].float(), target[:, :, 1:].float())
        reconstruction_loss = loss * self.scheduler.training_weight(timestep).to(loss.device)
        return reconstruction_loss + self.fsq_entropy_weight * entropy_loss
