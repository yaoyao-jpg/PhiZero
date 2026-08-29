#!/usr/bin/env python3
"""Extract tokenizer targets and write reasoner-ready ms-swift JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffsynth.pipelines.phizero import (
    PHIZERO_NUM_FRAMES,
    PHIZERO_OUTPUT_HEIGHT,
    PHIZERO_OUTPUT_WIDTH,
    PhiZeroPhysicalLanguageEncoder,
)


USER_PROMPT = '<image>\nText caption: {prompt}\nPredict the motion token sequence.'
EXPECTED_SYMBOLS = 256
EXPECTED_CODEBOOK_SIZE = 25_000
ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Sample:
    video: Path
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def load_samples(csv_path: Path) -> list[Sample]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV does not exist: {csv_path}')

    samples = []
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'video', 'prompt'}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{csv_path} is missing required columns: {sorted(missing)}')
        for line_number, row in enumerate(reader, start=2):
            raw_video = (row.get('video') or '').strip()
            prompt = (row.get('prompt') or '').strip()
            if not raw_video:
                raise ValueError(f'{csv_path}:{line_number} has an empty video path.')
            if not prompt:
                raise ValueError(f'{csv_path}:{line_number} has an empty prompt.')
            if '\n' in prompt or '\r' in prompt:
                raise ValueError(f'{csv_path}:{line_number} prompt must be one line.')
            video = Path(raw_video).expanduser()
            if not video.is_absolute():
                video = csv_path.parent / video
            video = video.resolve()
            if not video.is_file():
                raise FileNotFoundError(f'{csv_path}:{line_number} video does not exist: {video}')
            samples.append(Sample(video=video, prompt=prompt))
    if not samples:
        raise ValueError(f'{csv_path} contains no samples.')
    return samples


def load_video(path: Path) -> list[Image.Image]:
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            frames.append(Image.fromarray(frame).convert('RGB'))
            if len(frames) == PHIZERO_NUM_FRAMES:
                break
    finally:
        reader.close()
    if len(frames) != PHIZERO_NUM_FRAMES:
        raise ValueError(f'{path} has fewer than {PHIZERO_NUM_FRAMES} frames.')
    return frames


def assistant_target(indices: list[int]) -> str:
    return '<bom>' + ''.join(f'<m_{index}>' for index in indices) + '<eom>'


def repo_relative(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def jsonl_row(sample: Sample, first_frame: Path, indices: list[int]) -> dict:
    return {
        'messages': [
            {'role': 'user', 'content': USER_PROMPT.format(prompt=sample.prompt)},
            {'role': 'assistant', 'content': assistant_target(indices)},
        ],
        'images': [repo_relative(first_frame)],
    }


def save_first_frame(frame: Image.Image, path: Path) -> None:
    if frame.size != (PHIZERO_OUTPUT_WIDTH, PHIZERO_OUTPUT_HEIGHT):
        frame = frame.resize(
            (PHIZERO_OUTPUT_WIDTH, PHIZERO_OUTPUT_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )
    frame.save(path, format='JPEG', quality=95, subsampling=0)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.csv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Tokenizer checkpoint does not exist: {checkpoint}')

    output_dir = args.output_dir.expanduser().resolve()
    output_jsonl = output_dir / 'train.jsonl'
    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f'{output_jsonl} already exists; pass --overwrite to replace it.')
    first_frame_dir = output_dir / 'first_frames'
    first_frame_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA is unavailable; cannot use --device {args.device}.')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    encoder = PhiZeroPhysicalLanguageEncoder.from_pretrained(
        checkpoint,
        device=args.device,
    )
    model_config = encoder.model_config.motion_tokenizer
    symbol_count = model_config.physical_language_length(PHIZERO_NUM_FRAMES)
    if symbol_count != EXPECTED_SYMBOLS or model_config.codebook_size != EXPECTED_CODEBOOK_SIZE:
        raise ValueError(
            'The reasoner data contract requires a 25,000-code tokenizer that emits '
            f'256 symbols; this checkpoint emits {symbol_count} symbols with '
            f'{model_config.codebook_size} codes.'
        )

    temporary_jsonl = output_dir / '.train.jsonl.tmp'
    with temporary_jsonl.open('w', encoding='utf-8') as handle:
        for index, sample in enumerate(
            tqdm(samples, disable=args.no_progress, desc='Extracting Physical Language')
        ):
            frames = load_video(sample.video)
            indices = encoder.extract_physical_language(frames).tolist()
            if len(indices) != EXPECTED_SYMBOLS:
                raise RuntimeError(
                    f'{sample.video} produced {len(indices)} symbols; expected {EXPECTED_SYMBOLS}.'
                )
            first_frame = first_frame_dir / f'{index:08d}.jpg'
            save_first_frame(frames[0], first_frame)
            handle.write(
                json.dumps(
                    jsonl_row(sample, first_frame, indices),
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
                + '\n'
            )
    temporary_jsonl.replace(output_jsonl)
    print(f'Wrote {len(samples)} reasoner GT rows to {output_jsonl}')


if __name__ == '__main__':
    main()
