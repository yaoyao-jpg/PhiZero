# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from swift.utils import to_device

from .protocol import InferRequest
from .transformers_engine import TransformersEngine


PHYSICAL_LANGUAGE_PROMPT = '<image>\nText caption: {prompt}\nPredict the motion token sequence.'
PHYSICAL_LANGUAGE_LENGTH = 256
PHYSICAL_LANGUAGE_BEGIN_ID = 151_669
PHYSICAL_LANGUAGE_END_ID = 151_670
PHYSICAL_LANGUAGE_FIRST_SYMBOL_ID = 151_671
PHYSICAL_LANGUAGE_LAST_SYMBOL_ID = 176_670


def _patch_flash_attention_position_ids() -> None:
    import transformers.modeling_flash_attention_utils as flash_utils

    original = flash_utils._is_packed_sequence

    def is_packed_sequence(position_ids, *args, **kwargs):
        if position_ids is not None and position_ids.dim() > 2:
            return False
        return original(position_ids, *args, **kwargs)

    flash_utils._is_packed_sequence = is_packed_sequence


class PhysicalLanguageLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.symbol_ids = None

    @staticmethod
    def _allow(scores: torch.Tensor, token_ids) -> torch.Tensor:
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked[:, token_ids] = scores[:, token_ids]
        return masked

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids.shape[1] - self.prompt_length
        if generated == 0:
            return self._allow(scores, [PHYSICAL_LANGUAGE_BEGIN_ID])
        if generated <= PHYSICAL_LANGUAGE_LENGTH:
            if self.symbol_ids is None:
                self.symbol_ids = torch.arange(
                    PHYSICAL_LANGUAGE_FIRST_SYMBOL_ID,
                    PHYSICAL_LANGUAGE_LAST_SYMBOL_ID + 1,
                    device=scores.device,
                )
            return self._allow(scores, self.symbol_ids)
        return self._allow(scores, [PHYSICAL_LANGUAGE_END_ID])


class PhysicalLanguageReasoner:
    def __init__(self, checkpoint: str | Path) -> None:
        _patch_flash_attention_position_ids()
        self.engine = TransformersEngine(
            str(checkpoint),
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_impl='flash_attn',
            model_type='qwen3_vl',
            max_batch_size=1,
        )
        self.engine.template.max_length = 1024

    def _encode(self, image_path: str | Path, prompt: str) -> tuple[dict[str, Any], int]:
        request = InferRequest(
            messages=[{
                'role': 'user',
                'content': PHYSICAL_LANGUAGE_PROMPT.format(prompt=prompt.strip()),
            }],
            images=[str(Path(image_path).resolve())],
            chat_template_kwargs={'response_prefix': ''},
        )
        self.engine.model.eval()
        self.engine.template.set_mode('transformers')
        if self.engine.template.use_model:
            self.engine.template.model = self.engine.model
        encoded = self.engine.template.encode(request, return_template_inputs=True)
        encoded.pop('template_inputs', None)
        inputs = self.engine.template.data_collator([encoded])
        inputs = to_device(inputs, self.engine.model.device)
        _, inputs = self.engine.template.pre_forward_hook(self.engine.model, None, inputs)
        return inputs, int(self.engine._get_num_tokens(inputs))

    @torch.inference_mode()
    def predict(self, image_path: str | Path, prompt: str, seed: int = 1) -> list[int]:
        torch.manual_seed(seed)
        inputs, prompt_length = self._encode(image_path, prompt)
        generation_config = GenerationConfig(
            max_new_tokens=PHYSICAL_LANGUAGE_LENGTH + 2,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            pad_token_id=self.engine.tokenizer.pad_token_id,
            eos_token_id=self.engine.tokenizer.eos_token_id,
        )
        model_kwargs = {
            **inputs,
            'generation_config': generation_config,
            'logits_processor': LogitsProcessorList([
                PhysicalLanguageLogitsProcessor(prompt_length),
            ]),
        }
        model_kwargs = self.engine.template.prepare_generate_kwargs(
            model_kwargs,
            model=self.engine.model,
        )
        generated = self.engine.template.generate(self.engine.model, **model_kwargs)
        token_ids = self.engine.template.get_generate_ids(
            generated.sequences,
            prompt_length,
        )[0].tolist()
        return [token_id - PHYSICAL_LANGUAGE_FIRST_SYMBOL_ID for token_id in token_ids[1:-1]]
