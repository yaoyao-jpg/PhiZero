from __future__ import annotations

from typing import Sequence

import torch
from einops import rearrange
from torch import nn

from .config import PhiZeroModelConfig


class FiniteScalarCodeDecoder(nn.Module):
    def __init__(self, levels: tuple[int, ...], embedding_dim: int) -> None:
        super().__init__()
        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        basis = torch.cumprod(torch.tensor((1, *levels[:-1]), dtype=torch.int32), dim=0)
        self.register_buffer('levels', levels_tensor, persistent=False)
        self.register_buffer('basis', basis, persistent=False)
        self.output_projection = nn.Linear(len(levels), embedding_dim)

    def project_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codes = self.output_projection(codes.to(self.output_projection.weight.dtype))
        return rearrange(codes, 'b t d -> b d t 1 1')

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flattened = rearrange(indices, 'b ... -> b (...) 1')
        values = (flattened // self.basis) % self.levels
        half_width = self.levels // 2
        values = (values - half_width) / half_width
        return self.project_codes(values)


class PhysicalLanguageContextProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(tokens))


class PhysicalLanguageDecoderConditioner(nn.Module):
    """The physical-language tail used during video decoding."""

    def __init__(self, config: PhiZeroModelConfig) -> None:
        super().__init__()
        tokenizer = config.motion_tokenizer
        projector_hidden_dim = config.projector_hidden_dim or config.cross_attention_dim
        self.max_motion_tokens = config.max_motion_tokens
        self.codebook_size = tokenizer.codebook_size
        self.code_decoder = FiniteScalarCodeDecoder(tokenizer.levels, tokenizer.embedding_dim)
        self.post_quant_projection = nn.Conv3d(
            tokenizer.embedding_dim,
            tokenizer.z_channels,
            kernel_size=1,
        )
        self.context_projection = nn.Sequential(
            nn.Conv3d(tokenizer.z_channels, tokenizer.connector_out_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(
                tokenizer.connector_out_channels,
                tokenizer.connector_out_channels,
                kernel_size=1,
            ),
        )
        self.context_projector = PhysicalLanguageContextProjector(
            tokenizer.connector_out_channels,
            config.cross_attention_dim,
            projector_hidden_dim,
        )

    def _project_context(self, codes: torch.Tensor) -> torch.Tensor:
        context = self.context_projection(self.post_quant_projection(codes))
        context = context.flatten(2).permute(0, 2, 1)
        context = self.context_projector(context)
        return context[:, :self.max_motion_tokens] if self.max_motion_tokens is not None else context

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._project_context(self.code_decoder.project_codes(codes))

    def forward(self, indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        device = self.code_decoder.levels.device
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        indices = indices.reshape(indices.shape[0], -1, 1, 1)
        return self._project_context(self.code_decoder(indices))
