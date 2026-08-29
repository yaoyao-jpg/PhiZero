#!/usr/bin/env python3
"""Render the public PhiZero examples from physical-language sequences."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from diffsynth.pipelines.phizero import PhiZeroDiffusionDecoder
from diffsynth.utils.data import save_video


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / 'data/inference_cases/cases.json'
OUTPUT_DIR = ROOT / 'outputs/inference_cases'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dit-path', nargs='+')
    parser.add_argument('--vae-path')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cases = json.loads(CASES_PATH.read_text())
    if args.case:
        cases_by_id = {case['id']: case for case in cases}
        cases = [cases_by_id[case_id] for case_id in args.case]

    video_dir = OUTPUT_DIR / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        case for case in cases
        if args.overwrite or not (video_dir / f"{case['id']}.mp4").exists()
    ]

    if pending:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        decoder = PhiZeroDiffusionDecoder.from_pretrained(
            args.checkpoint,
            dit_path=args.dit_path,
            vae_path=args.vae_path,
            device='cuda:0',
            torch_dtype=torch.bfloat16,
        )

        for case in tqdm(pending, disable=args.no_progress, desc='Render'):
            image_path = CASES_PATH.parent / case['image']
            token_path = OUTPUT_DIR / 'physical_language' / f"{case['id']}.json"
            tokens = json.loads(token_path.read_text())
            with Image.open(image_path) as image:
                frames = decoder.render(
                    image.convert('RGB'),
                    tokens,
                    seed=1,
                    num_inference_steps=50,
                    progress=not args.no_progress,
                )
            save_video(frames, str(video_dir / f"{case['id']}.mp4"), fps=8, quality=5)

    print(f'Render: written={len(pending)}, skipped={len(cases) - len(pending)}')


if __name__ == '__main__':
    main()
