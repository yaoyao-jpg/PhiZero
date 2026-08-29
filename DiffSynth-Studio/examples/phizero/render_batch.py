#!/usr/bin/env python3
"""Render one Physics-IQ shard with the PhiZero decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
from tqdm.auto import tqdm

from diffsynth.pipelines.phizero import PhiZeroDiffusionDecoder


ROOT = Path(__file__).resolve().parents[3]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dit-path', nargs='+')
    parser.add_argument('--vae-path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()

    cases = json.loads(args.manifest.read_text())[args.rank::args.world_size]
    for case in cases:
        for key in ('image', 'tokens', 'video'):
            case[key] = repo_path(case[key])
    pending = [
        case for case in cases
        if args.overwrite or not Path(case['video']).exists()
    ]
    if pending:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        decoder = PhiZeroDiffusionDecoder.from_pretrained(
            args.checkpoint,
            dit_path=args.dit_path,
            vae_path=args.vae_path,
        )
        for case in tqdm(pending, disable=args.no_progress, desc=f'Render rank {args.rank}'):
            tokens = json.loads(Path(case['tokens']).read_text())
            with Image.open(case['image']) as image:
                frames = decoder.render(image, tokens, progress=not args.no_progress)
            frames = frames[:32] + [frames[32]] * 8
            video_path = Path(case['video'])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video = torch.from_numpy(
                np.stack([np.asarray(frame.convert('RGB'), dtype=np.uint8) for frame in frames])
            )
            write_video(
                str(video_path),
                video,
                fps=8,
                video_codec='libx264',
                options={'pix_fmt': 'yuv420p', 'crf': '23'},
            )

    print(f'Render rank {args.rank}: written={len(pending)}, skipped={len(cases) - len(pending)}')


if __name__ == '__main__':
    main()
