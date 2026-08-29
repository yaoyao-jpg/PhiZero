from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..wan_video_vae import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    Down_ResidualBlock,
    ResidualBlock,
    patchify,
)
from .config import PhysicalLanguageTokenizerConfig


class _PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(0)
        gaussian = torch.randn((2, embedding_dim // 2), generator=generator, device='cpu')
        self.register_buffer('gaussian', gaussian, persistent=False)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        grid = torch.ones((height, width), device=self.gaussian.device, dtype=self.gaussian.dtype)
        y = (grid.cumsum(dim=0) - 0.5) / height
        x = (grid.cumsum(dim=1) - 0.5) / width
        coordinates = 2 * torch.stack((x, y), dim=-1) - 1
        coordinates = 2 * math.pi * (coordinates @ self.gaussian)
        return torch.cat((torch.sin(coordinates), torch.cos(coordinates)), dim=-1).permute(2, 0, 1)


class _Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        tensor = tensor.reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        return tensor.transpose(1, 2)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = self._split_heads(self.q_proj(queries))
        keys = self._split_heads(self.k_proj(keys))
        values = self._split_heads(self.v_proj(values))
        attention = torch.softmax(queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1]), dim=-1)
        output = (attention @ values).transpose(1, 2)
        output = output.reshape(queries.shape[0], -1, self.internal_dim)
        return self.out_proj(output)


class _MLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.relu(self.lin1(tensor)))


class _QFormerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.cross_attn_token_to_image = _Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = _MLP(embedding_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.cross_attn_token_to_image(queries, keys + position, keys)
        queries = self.norm2(queries + attention)
        return self.norm3(queries + self.mlp(queries))


class _QFormer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _QFormerBlock(embedding_dim, num_heads, mlp_dim) for _ in range(depth)
        )
        self.final_attn_token_to_image = _Attention(embedding_dim, num_heads)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_position: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        keys = image_embedding.flatten(2).permute(0, 2, 1)
        position = image_position.flatten(2).permute(0, 2, 1)
        for layer in self.layers:
            queries = layer(queries, keys, position)
        attention = self.final_attn_token_to_image(queries + queries, keys + position, keys)
        return self.norm_final_attn(queries + attention)


class _AdjacentFrameQFormer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.qformer = _QFormer(depth, embedding_dim, num_heads, embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_position: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for index in range(image_embedding.shape[2] - 1):
            frame_slice = slice(index, index + 2)
            outputs.append(
                self.qformer(
                    image_embedding[:, :, frame_slice],
                    image_position[:, :, frame_slice],
                    queries,
                )
            )
        return torch.cat(outputs, dim=1)


class _PhysicalVideoEncoder(nn.Module):
    def __init__(self, config: PhysicalLanguageTokenizerConfig) -> None:
        super().__init__()
        dimensions = (config.dim, *(config.dim * value for value in config.dim_mult))
        self.conv1 = CausalConv3d(config.in_channels * 4, dimensions[0], 3, padding=1)
        self.downsamples = nn.Sequential(
            *(
                Down_ResidualBlock(
                    in_dim=input_dim,
                    out_dim=output_dim,
                    dropout=config.dropout,
                    mult=config.num_res_blocks,
                    temperal_downsample=(
                        config.temperal_downsample[index]
                        if index < len(config.temperal_downsample)
                        else False
                    ),
                    down_flag=index < len(config.dim_mult) - 1,
                )
                for index, (input_dim, output_dim) in enumerate(
                    zip(dimensions[:-1], dimensions[1:])
                )
            )
        )
        output_dim = dimensions[-1]
        self.middle = nn.Sequential(
            ResidualBlock(output_dim, output_dim, config.dropout),
            AttentionBlock(output_dim),
            ResidualBlock(output_dim, output_dim, config.dropout),
        )


class _FiniteScalarCodeEncoder(nn.Module):
    def __init__(self, levels: tuple[int, ...], embedding_dim: int) -> None:
        super().__init__()
        levels_tensor = torch.tensor(levels, dtype=torch.int32, device='cpu')
        basis = torch.cumprod(
            torch.tensor((1, *levels[:-1]), dtype=torch.int32, device='cpu'),
            dim=0,
        )
        self.register_buffer('levels', levels_tensor, persistent=False)
        self.register_buffer('basis', basis, persistent=False)
        self.input_projection = nn.Linear(embedding_dim, len(levels))

    def _bound(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(tokens.to(self.input_projection.weight.dtype))
        levels = self.levels.to(tokens.device)
        half_levels = (levels - 1).to(tokens.dtype) * ((1 + 1e-3) / 2)
        offsets = torch.where(
            levels % 2 == 0,
            tokens.new_full(levels.shape, 0.5),
            tokens.new_zeros(levels.shape),
        )
        return (tokens + (offsets / half_levels).atanh()).tanh() * half_levels - offsets

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = self._bound(tokens)
        levels = self.levels.to(tokens.device)
        half_width = (levels // 2).to(bounded.dtype)
        rounded = bounded + (bounded.round() - bounded).detach()
        codes = rounded / half_width
        values = (codes * half_width + half_width).float()
        indices = (values * self.basis.to(tokens.device)).sum(dim=-1).to(torch.int32)
        return indices, codes

    def soft_entropy_loss(
        self,
        tokens: torch.Tensor,
        *,
        temperature: float = 0.1,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        bounded = self._bound(tokens).float()
        probabilities = []
        sample_entropies = []
        for index, level in enumerate(self.levels.tolist()):
            half_width = level // 2
            scalar = bounded[..., index] / half_width
            values = (torch.arange(level, device=tokens.device) - half_width) / half_width
            logits = -(
                (scalar[..., None] - values) ** 2 / temperature**2
            )
            probs = torch.softmax(logits, dim=-1)
            probabilities.append(probs.flatten(0, 1))
            sample_entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(-1).mean())

        joint = probabilities[0]
        for probs in probabilities[1:]:
            joint = (joint.unsqueeze(-1) * probs.unsqueeze(-2)).flatten(-2)
        marginal = joint.mean(dim=0)
        batch_entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum()
        normalizer = sum(math.log(level) for level in self.levels.tolist())
        return (torch.stack(sample_entropies).sum() - gamma * batch_entropy) / normalizer

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        indices, _ = self.encode(tokens)
        return indices


class PhysicalLanguageTokenizer(nn.Module):
    """Encode video transitions as discrete physical-language symbols."""

    def __init__(self, config: PhysicalLanguageTokenizerConfig) -> None:
        super().__init__()
        if config.vae_arch != 'wan2.2_vae38':
            raise ValueError(f'Unsupported tokenizer VAE: {config.vae_arch}')
        if config.qformer_type != 'QFormerAdjacentFSingleQ':
            raise ValueError(f'Unsupported QFormer: {config.qformer_type}')
        if config.quantizer.lower() != 'fsq':
            raise ValueError(f'Unsupported quantizer: {config.quantizer}')

        self.config = config
        self.patch_size = 2
        self.encoder_chunk_size = 2 ** sum(config.temperal_downsample)
        self.video_encoder = _PhysicalVideoEncoder(config)
        self.position_embedding = _PositionEmbedding(config.hidden_dim)
        self.learned_queries = nn.Embedding(config.act_embedding_num, config.hidden_dim)
        self.motion_queries = _AdjacentFrameQFormer(
            config.qformer_depth,
            config.hidden_dim,
            config.qformer_num_heads,
        )
        self.token_projection = nn.Conv3d(config.hidden_dim, config.z_channels, kernel_size=1)
        self.fsq_projection = nn.Conv3d(
            config.z_channels,
            config.embedding_dim,
            kernel_size=1,
        )
        self.quantizer = _FiniteScalarCodeEncoder(config.levels, config.embedding_dim)

    @staticmethod
    def _causal_conv(
        layer: CausalConv3d,
        tensor: torch.Tensor,
        cache: list[torch.Tensor | str | None],
        index: list[int],
    ) -> torch.Tensor:
        cache_index = index[0]
        cached_input = tensor[:, :, -CACHE_T:].clone()
        previous = cache[cache_index]
        if cached_input.shape[2] < CACHE_T and isinstance(previous, torch.Tensor):
            cached_input = torch.cat((previous[:, :, -1:].to(tensor.device), cached_input), dim=2)
        tensor = layer(tensor, previous)
        cache[cache_index] = cached_input
        index[0] += 1
        return tensor

    def _encoder_chunk(
        self,
        tensor: torch.Tensor,
        cache: list[torch.Tensor | str | None],
    ) -> torch.Tensor:
        index = [0]
        tensor = self._causal_conv(self.video_encoder.conv1, tensor, cache, index)
        for module in self.video_encoder.downsamples:
            tensor, cache, index = module(tensor, cache, index)
        for module in self.video_encoder.middle:
            if isinstance(module, ResidualBlock):
                tensor, cache, index = module(tensor, cache, index)
            else:
                tensor = module(tensor)
        return tensor

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        video = patchify(video, self.patch_size)
        cache = [None] * sum(
            isinstance(module, CausalConv3d) for module in self.video_encoder.modules()
        )
        chunks = []
        for step in range(1 + (video.shape[2] - 1) // self.encoder_chunk_size):
            if step == 0:
                chunk = video[:, :, :1]
            else:
                start = 1 + self.encoder_chunk_size * (step - 1)
                chunk = video[:, :, start:start + self.encoder_chunk_size]
            chunks.append(self._encoder_chunk(chunk, cache))
        return torch.cat(chunks, dim=2)

    def _token_features(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self._encode_video(video)
        batch, _, frames, height, width = hidden.shape
        position = self.position_embedding((height, width)).to(hidden)
        position = position.unsqueeze(1).repeat(1, frames, 1, 1)[None].repeat(batch, 1, 1, 1, 1)
        queries = self.learned_queries.weight.to(hidden)[None].repeat(batch, 1, 1)
        queries = self.motion_queries(hidden, position, queries)
        queries = self.token_projection(queries.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1))
        tokens = self.fsq_projection(queries)
        return rearrange(tokens, 'b d t h w -> b (t h w) d')

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantizer.encode(self._token_features(video))

    def encode_for_training(
        self,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self._token_features(video)
        indices, codes = self.quantizer.encode(tokens)
        return indices, codes, self.quantizer.soft_entropy_loss(tokens)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        indices, _ = self.encode(video)
        return indices
