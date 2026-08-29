from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open
from torch import nn


_CONDITIONER_KEY_MAP = (
    (
        'motion_conditioner.motion_tokenizer.quantizer.project_out.',
        'code_decoder.output_projection.',
    ),
    (
        'motion_conditioner.motion_tokenizer.post_quant_conv.',
        'post_quant_projection.',
    ),
    (
        'motion_conditioner.motion_tokenizer.quant_to_dit_dim.',
        'context_projection.',
    ),
    (
        'motion_conditioner.cross_attention_projector.',
        'context_projector.',
    ),
)

_TOKENIZER_KEY_MAP = (
    (
        'motion_conditioner.motion_tokenizer.model.encoder.',
        'video_encoder.',
    ),
    (
        'motion_conditioner.motion_tokenizer.act.',
        'learned_queries.',
    ),
    (
        'motion_conditioner.motion_tokenizer.qformer_ldm.',
        'motion_queries.',
    ),
    (
        'motion_conditioner.motion_tokenizer.outproj_ldm.',
        'token_projection.',
    ),
    (
        'motion_conditioner.motion_tokenizer.quant_conv.',
        'fsq_projection.',
    ),
    (
        'motion_conditioner.motion_tokenizer.quantizer.project_in.',
        'quantizer.input_projection.',
    ),
)

_LORA_A_SUFFIX = '.lora_A.default.weight'
_LORA_B_SUFFIX = '.lora_B.default.weight'


def _mapped_state_dict(handle, key_map, *, skip_prefix: str | None = None):
    state_dict = {}
    for key in handle.keys():
        if skip_prefix is not None and key.startswith(skip_prefix):
            continue
        for source_prefix, target_prefix in key_map:
            if key.startswith(source_prefix):
                state_dict[target_prefix + key.removeprefix(source_prefix)] = handle.get_tensor(key)
                break
    return state_dict


def _lora_module_names(handle) -> tuple[str, ...]:
    prefix = 'dit.'
    return tuple(
        key.removeprefix(prefix).removesuffix(_LORA_A_SUFFIX)
        for key in handle.keys()
        if key.startswith(prefix) and key.endswith(_LORA_A_SUFFIX)
    )


def load_physical_language_decoder(conditioner: nn.Module, checkpoint: str | Path) -> None:
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        state_dict = _mapped_state_dict(handle, _CONDITIONER_KEY_MAP)
    conditioner.load_state_dict(state_dict, strict=True, assign=True)


@torch.no_grad()
def load_phi_zero_weights(
    conditioner: nn.Module,
    dit: nn.Module,
    checkpoint: str | Path,
) -> None:
    """Load the decoder tail and merge the checkpoint LoRA into the base DiT."""
    load_physical_language_decoder(conditioner, checkpoint)
    modules = dict(dit.named_modules())
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        for name in _lora_module_names(handle):
            weight = modules[name].weight
            a = handle.get_tensor(f'dit.{name}{_LORA_A_SUFFIX}').to(weight)
            b = handle.get_tensor(f'dit.{name}{_LORA_B_SUFFIX}').to(weight)
            weight.add_(torch.mm(b, a))


def load_phi_zero_lora(dit: nn.Module, checkpoint: str | Path) -> tuple[str, ...]:
    """Inject trainable PEFT adapters and initialize them from a PhiZero checkpoint."""
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        module_names = _lora_module_names(handle)
        rank = handle.get_tensor(f'dit.{module_names[0]}{_LORA_A_SUFFIX}').shape[0]

    dit.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=list(module_names),
        lora_dropout=0.0,
        bias='none',
    )
    inject_adapter_in_model(config, dit)

    modules = dict(dit.named_modules())
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        for name in module_names:
            module = modules[name]
            base_weight = module.base_layer.weight
            module.lora_A['default'].weight.data = handle.get_tensor(
                f'dit.{name}{_LORA_A_SUFFIX}'
            ).to(base_weight)
            module.lora_B['default'].weight.data = handle.get_tensor(
                f'dit.{name}{_LORA_B_SUFFIX}'
            ).to(base_weight)
    return module_names


def load_physical_language_tokenizer(tokenizer: nn.Module, checkpoint: str | Path) -> None:
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        state_dict = _mapped_state_dict(
            handle,
            _TOKENIZER_KEY_MAP,
            skip_prefix='motion_conditioner.motion_tokenizer.model.encoder.head.',
        )
    tokenizer.load_state_dict(state_dict, strict=True, assign=True)


def _export_component(module: nn.Module, key_map) -> dict[str, torch.Tensor]:
    state_dict = {}
    for key, value in module.state_dict().items():
        for source_prefix, target_prefix in key_map:
            if key.startswith(target_prefix):
                output_key = source_prefix + key.removeprefix(target_prefix)
                state_dict[output_key] = value.detach().to(device='cpu').contiguous()
                break
    return state_dict


def export_phi_zero_state_dict(
    tokenizer: nn.Module,
    conditioner: nn.Module,
    dit: nn.Module,
    lora_module_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Export a checkpoint that can be loaded directly by the release inference code."""
    state_dict = _export_component(tokenizer, _TOKENIZER_KEY_MAP)
    state_dict.update(_export_component(conditioner, _CONDITIONER_KEY_MAP))
    modules = dict(dit.named_modules())
    for name in lora_module_names:
        module = modules[name]
        state_dict[f'dit.{name}{_LORA_A_SUFFIX}'] = (
            module.lora_A['default'].weight.detach().to(device='cpu').contiguous()
        )
        state_dict[f'dit.{name}{_LORA_B_SUFFIX}'] = (
            module.lora_B['default'].weight.detach().to(device='cpu').contiguous()
        )
    return state_dict
