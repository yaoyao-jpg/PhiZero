from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhysicalLanguageTokenizerConfig:
    in_channels: int
    dim: int
    z_channels: int
    vae_arch: str
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    attn_scales: tuple[float, ...]
    temperal_downsample: tuple[bool, ...]
    dropout: float
    embedding_dim: int
    quantizer: str
    levels: tuple[int, ...]
    num_embeddings: int | None
    beta: float
    persistent_quantizer: bool
    act_embedding_num: int
    qformer_type: str
    qformer_depth: int
    qformer_num_heads: int
    connector_out_channels: int
    name: str

    @classmethod
    def from_dict(cls, values: dict) -> 'PhysicalLanguageTokenizerConfig':
        values = dict(values)
        for field in ('dim_mult', 'attn_scales', 'temperal_downsample', 'levels'):
            values[field] = tuple(values[field])
        return cls(**values)

    @property
    def hidden_dim(self) -> int:
        return self.dim * self.dim_mult[-1]

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)

    def physical_language_length(self, source_frames: int) -> int:
        encoded_frames = source_frames
        for downsample in self.temperal_downsample:
            if downsample:
                encoded_frames = 1 + (encoded_frames - 1) // 2
        return (encoded_frames - 1) * self.act_embedding_num


@dataclass(frozen=True)
class PhiZeroModelConfig:
    motion_tokenizer: PhysicalLanguageTokenizerConfig
    cross_attention_dim: int
    projector_hidden_dim: int | None
    max_motion_tokens: int | None

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> 'PhiZeroModelConfig':
        config_path = Path(checkpoint).parent / 'model_config.json'
        values = json.loads(config_path.read_text())
        return cls(
            motion_tokenizer=PhysicalLanguageTokenizerConfig.from_dict(
                values['motion_tokenizer_config']
            ),
            cross_attention_dim=values['cross_attention_dim'],
            projector_hidden_dim=values['projector_hidden_dim'],
            max_motion_tokens=values['max_motion_tokens'],
        )
