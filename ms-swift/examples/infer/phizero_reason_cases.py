#!/usr/bin/env python3
"""Predict physical-language sequences for the public PhiZero examples."""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from swift.infer_engine import PhysicalLanguageReasoner


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / 'data/inference_cases/cases.json'
OUTPUT_DIR = ROOT / 'outputs/inference_cases'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
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

    output_dir = OUTPUT_DIR / 'physical_language'
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        case for case in cases
        if args.overwrite or not (output_dir / f"{case['id']}.json").exists()
    ]

    if pending:
        reasoner = PhysicalLanguageReasoner(args.model)

        for case in tqdm(pending, disable=args.no_progress, desc='Physical language'):
            image = CASES_PATH.parent / case['image']
            tokens = reasoner.predict(image, case['prompt'])
            path = output_dir / f"{case['id']}.json"
            path.write_text(json.dumps(tokens) + '\n')

    print(f'Physical language: written={len(pending)}, skipped={len(cases) - len(pending)}')


if __name__ == '__main__':
    main()
