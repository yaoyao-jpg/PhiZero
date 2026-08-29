#!/usr/bin/env python3
"""Predict one Physics-IQ shard with the PhiZero reasoner."""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from swift.infer_engine import PhysicalLanguageReasoner


ROOT = Path(__file__).resolve().parents[3]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()

    cases = json.loads(args.manifest.read_text())[args.rank::args.world_size]
    for case in cases:
        for key in ('source_image', 'image', 'tokens'):
            case[key] = repo_path(case[key])
    pending = [
        case for case in cases
        if args.overwrite or not Path(case['tokens']).exists()
    ]
    if pending:
        reasoner = PhysicalLanguageReasoner(args.model)
        for case in tqdm(pending, disable=args.no_progress, desc=f'Reason rank {args.rank}'):
            image_path = Path(case['image'])
            if not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(case['source_image']) as image:
                    image.convert('RGB').resize((896, 512), Image.Resampling.LANCZOS).save(
                        image_path,
                        quality=95,
                    )
            tokens = reasoner.predict(image_path, case['prompt'])
            token_path = Path(case['tokens'])
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(json.dumps(tokens) + '\n')

    print(f'Reason rank {args.rank}: written={len(pending)}, skipped={len(cases) - len(pending)}')


if __name__ == '__main__':
    main()
