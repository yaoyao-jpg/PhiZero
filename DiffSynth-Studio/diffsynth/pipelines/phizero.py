from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageOps
from torch import nn
from tqdm.auto import tqdm

from ..core import ModelConfig
from ..core.gradient import gradient_checkpoint_forward
from ..diffusion import FlowMatchScheduler
from ..models.model_loader import ModelPool
from ..models.phizero import (
    PhiZeroModelConfig,
    PhysicalLanguageDecoderConditioner,
    PhysicalLanguageTokenizer,
    load_phi_zero_weights,
    load_physical_language_tokenizer,
)
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE38


PHIZERO_OUTPUT_WIDTH = 896
PHIZERO_OUTPUT_HEIGHT = 512
PHIZERO_FPS = 8
PHIZERO_NUM_FRAMES = 33
PHIZERO_TOKENIZER_WIDTH = 448
PHIZERO_TOKENIZER_HEIGHT = 256


def predict_physical_language_flow(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    *,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Run the PhiZero-conditioned Wan flow predictor."""
    hidden = dit.patchify(latents)
    frames, height, width = hidden.shape[2:]
    timestep_scalar = timestep.flatten()[0]
    tokens_per_frame = height * width
    separated_timestep = torch.cat(
        (
            torch.zeros(tokens_per_frame, dtype=latents.dtype, device=latents.device),
            torch.ones(
                (latents.shape[2] - 1) * tokens_per_frame,
                dtype=latents.dtype,
                device=latents.device,
            ) * timestep_scalar,
        )
    )
    time_embedding = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, separated_timestep)
        .unsqueeze(0)
        .to(dtype=latents.dtype)
    )
    time_modulation = dit.time_projection(time_embedding).unflatten(2, (6, dit.dim))

    hidden = rearrange(hidden, 'b c f h w -> b (f h w) c').contiguous()
    frequencies = torch.cat(
        (
            dit.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            dit.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            dit.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ),
        dim=-1,
    ).reshape(frames * height * width, 1, -1).to(hidden.device)
    for block in dit.blocks:
        hidden = gradient_checkpoint_forward(
            block,
            gradient_checkpointing,
            False,
            hidden,
            context,
            time_modulation,
            frequencies,
        )
    hidden = dit.head(hidden, time_embedding)
    return dit.unpatchify(hidden, (frames, height, width))


def load_wan_base_models(
    device: str | torch.device,
    torch_dtype: torch.dtype,
    dit_path: str | Sequence[str] | None,
    vae_path: str | None,
) -> tuple[WanModel, WanVideoVAE38]:
    configs = [
        ModelConfig(
            path=dit_path,
            model_id=None if dit_path else 'Wan-AI/Wan2.2-TI2V-5B',
            origin_file_pattern=None if dit_path else 'diffusion_pytorch_model*.safetensors',
        ),
        ModelConfig(
            path=vae_path,
            model_id=None if vae_path else 'DiffSynth-Studio/Wan-Series-Converted-Safetensors',
            origin_file_pattern=None if vae_path else 'Wan2.2_VAE.safetensors',
        ),
    ]
    pool = ModelPool()
    for config in configs:
        config.download_if_necessary()
        vram_config = pool.default_vram_config()
        vram_config.update(
            computation_device=device,
            computation_dtype=torch_dtype,
            onload_device=device,
            onload_dtype=torch_dtype,
        )
        pool.auto_load_model(config.path, vram_config=vram_config)
    return pool.fetch_model('wan_video_dit'), pool.fetch_model('wan_video_vae')


class PhiZeroDiffusionDecoder(nn.Module):
    def __init__(
        self,
        dit: WanModel,
        vae: WanVideoVAE38,
        conditioner: PhysicalLanguageDecoderConditioner,
        model_config: PhiZeroModelConfig,
        device: str | torch.device,
        torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.dit = dit.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False)
        self.conditioner = conditioner.eval().requires_grad_(False)
        self.model_config = model_config
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.scheduler = FlowMatchScheduler('Wan')

    @classmethod
    def from_pretrained(
        cls,
        decoder_checkpoint: str | Path,
        *,
        dit_path: str | Sequence[str] | None = None,
        vae_path: str | None = None,
        device: str | torch.device = 'cuda:0',
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> 'PhiZeroDiffusionDecoder':
        model_config = PhiZeroModelConfig.from_checkpoint(decoder_checkpoint)
        dit, vae = load_wan_base_models(device, torch_dtype, dit_path, vae_path)
        conditioner = PhysicalLanguageDecoderConditioner(model_config)
        load_phi_zero_weights(conditioner, dit, decoder_checkpoint)
        conditioner = conditioner.to(device=device, dtype=torch_dtype)
        return cls(dit, vae, conditioner, model_config, device, torch_dtype)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.tensor(array, dtype=self.torch_dtype, device=self.device)
        return (tensor * (2.0 / 255.0) - 1.0).permute(2, 0, 1).unsqueeze(1)

    def _noise(self, shape: tuple[int, ...], seed: int) -> torch.Tensor:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        return noise.to(device=self.device, dtype=self.torch_dtype)

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return predict_physical_language_flow(self.dit, latents, timestep, context)

    @staticmethod
    def _to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
        video = video[0].permute(1, 2, 3, 0)
        video = ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(frame) for frame in video]

    @torch.inference_mode()
    def render(
        self,
        first_frame: Image.Image,
        physical_language: Sequence[int] | torch.Tensor,
        *,
        seed: int = 1,
        num_inference_steps: int = 50,
        progress: bool = True,
    ) -> list[Image.Image]:
        indices = torch.as_tensor(physical_language)
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        token_count = self.model_config.motion_tokenizer.physical_language_length(
            PHIZERO_NUM_FRAMES
        )
        if indices.shape != (token_count,) or indices.dtype not in integer_dtypes:
            raise ValueError(f'Physical language must contain exactly {token_count} integer indices.')
        if indices.min() < 0 or indices.max() >= self.conditioner.codebook_size:
            raise ValueError(
                f'Physical-language indices must be in [0, {self.conditioner.codebook_size}).'
            )

        first_frame = ImageOps.fit(
            first_frame.convert('RGB'),
            (PHIZERO_OUTPUT_WIDTH, PHIZERO_OUTPUT_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
        context = self.conditioner(indices).to(device=self.device, dtype=self.torch_dtype)
        temporal_factor = 2 ** sum(self.vae.model.temperal_upsample)
        latent_frames = 1 + (PHIZERO_NUM_FRAMES - 1) // temporal_factor
        latents = self._noise(
            (
                1,
                self.vae.z_dim,
                latent_frames,
                PHIZERO_OUTPUT_HEIGHT // self.vae.upsampling_factor,
                PHIZERO_OUTPUT_WIDTH // self.vae.upsampling_factor,
            ),
            seed,
        )
        first_frame_latents = self.vae.encode(
            [self._image_to_tensor(first_frame)],
            device=self.device,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents[:, :, :1] = first_frame_latents

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        timesteps: Iterable[torch.Tensor] = self.scheduler.timesteps
        if progress:
            timesteps = tqdm(timesteps, desc='Rendering future video')
        for step, timestep in enumerate(timesteps):
            model_timestep = timestep.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            noise = self._predict_noise(latents, model_timestep, context)
            latents = self.scheduler.step(noise, self.scheduler.timesteps[step], latents)
            latents[:, :, :1] = first_frame_latents

        video = self.vae.decode(latents, device=self.device)
        return self._to_pil_frames(video)


class PhiZeroPhysicalLanguageEncoder(nn.Module):
    """Tokenizer-only pipeline for extracting Physical Language ground truth."""

    def __init__(
        self,
        tokenizer: PhysicalLanguageTokenizer,
        model_config: PhiZeroModelConfig,
        device: str | torch.device,
        torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer.eval().requires_grad_(False)
        self.model_config = model_config
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_checkpoint: str | Path,
        *,
        device: str | torch.device = 'cuda:0',
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> 'PhiZeroPhysicalLanguageEncoder':
        model_config = PhiZeroModelConfig.from_checkpoint(tokenizer_checkpoint)
        with torch.device('meta'):
            tokenizer = PhysicalLanguageTokenizer(model_config.motion_tokenizer)
        load_physical_language_tokenizer(tokenizer, tokenizer_checkpoint)
        tokenizer = tokenizer.to(device=device, dtype=torch_dtype)
        return cls(tokenizer, model_config, device, torch_dtype)

    @torch.inference_mode()
    def extract_physical_language(self, source_frames: Sequence[Image.Image]) -> torch.Tensor:
        if len(source_frames) != PHIZERO_NUM_FRAMES:
            raise ValueError(
                f'Physical Language extraction expects {PHIZERO_NUM_FRAMES} frames; '
                f'received {len(source_frames)}.'
            )
        frame_sizes = {frame.size for frame in source_frames}
        if len(frame_sizes) != 1:
            raise ValueError(
                f'All source frames must have one resolution; received {frame_sizes}.'
            )

        frames = [
            torch.from_numpy(np.array(frame.convert('RGB'), dtype=np.float32, copy=True)).permute(
                2, 0, 1
            )
            for frame in source_frames
        ]
        video = torch.stack(frames, dim=1).unsqueeze(0)
        video = (video * (2.0 / 255.0) - 1.0).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        if video.shape[-2:] != (PHIZERO_OUTPUT_HEIGHT, PHIZERO_OUTPUT_WIDTH):
            video = F.interpolate(
                video,
                size=(PHIZERO_OUTPUT_HEIGHT, PHIZERO_OUTPUT_WIDTH),
                mode='bilinear',
                align_corners=False,
            )
        video = F.interpolate(
            video,
            size=(PHIZERO_TOKENIZER_HEIGHT, PHIZERO_TOKENIZER_WIDTH),
            mode='bilinear',
            align_corners=False,
        )
        video = rearrange(
            video,
            '(b t) c h w -> b c t h w',
            b=1,
            t=PHIZERO_NUM_FRAMES,
        )
        indices = self.tokenizer(video).reshape(-1).to(dtype=torch.int64, device='cpu')
        expected_count = self.model_config.motion_tokenizer.physical_language_length(
            PHIZERO_NUM_FRAMES
        )
        if indices.shape != (expected_count,):
            raise RuntimeError(
                f'Tokenizer returned {indices.numel()} symbols; expected {expected_count}.'
            )
        codebook_size = self.model_config.motion_tokenizer.codebook_size
        if indices.min() < 0 or indices.max() >= codebook_size:
            raise RuntimeError(f'Tokenizer returned an index outside [0, {codebook_size}).')
        return indices


class PhiZeroMotionTransferPipeline:
    def __init__(
        self,
        decoder: PhiZeroDiffusionDecoder,
        tokenizer: PhysicalLanguageTokenizer,
    ) -> None:
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.physical_language_encoder = PhiZeroPhysicalLanguageEncoder(
            tokenizer,
            decoder.model_config,
            decoder.device,
            decoder.torch_dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        decoder_checkpoint: str | Path,
        *,
        dit_path: str | Sequence[str] | None = None,
        vae_path: str | None = None,
        device: str | torch.device = 'cuda:0',
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> 'PhiZeroMotionTransferPipeline':
        decoder = PhiZeroDiffusionDecoder.from_pretrained(
            decoder_checkpoint,
            dit_path=dit_path,
            vae_path=vae_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        with torch.device('meta'):
            tokenizer = PhysicalLanguageTokenizer(decoder.model_config.motion_tokenizer)
        load_physical_language_tokenizer(tokenizer, decoder_checkpoint)
        tokenizer = tokenizer.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
        return cls(decoder, tokenizer)

    @torch.inference_mode()
    def extract_physical_language(self, source_frames: Sequence[Image.Image]) -> torch.Tensor:
        return self.physical_language_encoder.extract_physical_language(source_frames)

    def transfer(
        self,
        source_frames: Sequence[Image.Image],
        target_first_frame: Image.Image,
        *,
        seed: int = 1,
        num_inference_steps: int = 50,
        progress: bool = True,
    ) -> tuple[torch.Tensor, list[Image.Image]]:
        physical_language = self.extract_physical_language(source_frames)
        video = self.decoder.render(
            target_first_frame,
            physical_language,
            seed=seed,
            num_inference_steps=num_inference_steps,
            progress=progress,
        )
        return physical_language, video


class PhiZeroReconstructionPipeline(PhiZeroMotionTransferPipeline):
    """Reconstruct a video through the Physical Language bottleneck."""

    def reconstruct(
        self,
        source_frames: Sequence[Image.Image],
        *,
        seed: int = 1,
        num_inference_steps: int = 50,
        progress: bool = True,
    ) -> tuple[torch.Tensor, list[Image.Image]]:
        return self.transfer(
            source_frames,
            source_frames[0],
            seed=seed,
            num_inference_steps=num_inference_steps,
            progress=progress,
        )
