#!/usr/bin/env python3
"""Transfer physical language from a source video to a target first frame."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffsynth.pipelines.phizero import (
    PHIZERO_FPS,
    PHIZERO_NUM_FRAMES,
    PhiZeroMotionTransferPipeline,
)
from diffsynth.utils.data import save_video


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / 'data/motion_transfer/cases.json'
OUTPUT_DIR = ROOT / 'outputs/motion_transfer'


def load_source_video(path: Path) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    try:
        frames = [Image.fromarray(frame).convert('RGB') for frame in reader]
    finally:
        reader.close()
    if len(frames) != PHIZERO_NUM_FRAMES:
        raise ValueError(
            f'{path} contains {len(frames)} frames; PhiZero expects {PHIZERO_NUM_FRAMES}.'
        )
    return frames


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dit-path', nargs='+')
    parser.add_argument('--vae-path')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cases = json.loads(CASES_PATH.read_text())
    if args.case:
        cases_by_id = {case['id']: case for case in cases}
        cases = [cases_by_id[case_id] for case_id in args.case]
    if args.list_cases:
        print('\n'.join(case['id'] for case in cases))
        return

    token_dir = OUTPUT_DIR / 'physical_language'
    video_dir = OUTPUT_DIR / 'videos'
    token_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        case
        for case in cases
        if args.overwrite or not (video_dir / f"{case['id']}.mp4").exists()
    ]

    if pending:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        pipeline = PhiZeroMotionTransferPipeline.from_pretrained(
            args.checkpoint,
            dit_path=args.dit_path,
            vae_path=args.vae_path,
        )
        for case in tqdm(pending, disable=args.no_progress, desc='Motion transfer'):
            source_path = CASES_PATH.parent / case['source_video']
            target_path = CASES_PATH.parent / case['target_image']
            source_frames = load_source_video(source_path)
            with Image.open(target_path) as target_image:
                tokens, frames = pipeline.transfer(
                    source_frames,
                    target_image.convert('RGB'),
                    progress=not args.no_progress,
                )
            (token_dir / f"{case['id']}.json").write_text(json.dumps(tokens.tolist()))
            save_video(frames, str(video_dir / f"{case['id']}.mp4"), fps=PHIZERO_FPS, quality=5)

    print(f'Motion transfer: written={len(pending)}, skipped={len(cases) - len(pending)}')


if __name__ == '__main__':
    main()
