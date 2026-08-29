#!/usr/bin/env python3
"""Run PhiZero on Physics-IQ Verified with separate model environments."""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
DESCRIPTIONS = BENCHMARK_DIR / 'descriptions_phizero.csv'
REASONER = ROOT / 'ms-swift/examples/infer/phizero_reason_batch.py'
DECODER = ROOT / 'DiffSynth-Studio/examples/phizero/render_batch.py'
MODEL_NAME = 'phizero_checkpoint6993_decoder_step100000_seed1_physics_iq_verified'


def repo_relative(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def sanitize_outputs(output_root: Path) -> None:
    replacements = (
        (f'{ROOT}{os.sep}', './'),
        (f'{ROOT.parent}{os.sep}', '../'),
    )
    for pattern in ('*.json', '*.log', '*.txt', '*.csv'):
        for path in output_root.rglob(pattern):
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            sanitized = text
            for source, target in replacements:
                sanitized = sanitized.replace(source, target)
            if sanitized != text:
                path.write_text(sanitized, encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reason', 'decode', 'generate', 'score', 'all'), default='all')
    parser.add_argument('--benchmark-base', type=Path, required=True)
    parser.add_argument('--reasoner-python', required=True)
    parser.add_argument('--decoder-python', required=True)
    parser.add_argument('--score-python')
    parser.add_argument('--reasoner-checkpoint', required=True)
    parser.add_argument('--decoder-checkpoint', required=True)
    parser.add_argument('--dit-path', nargs='+')
    parser.add_argument('--vae-path')
    parser.add_argument('--output-root', type=Path, default=BENCHMARK_DIR / 'outputs')
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--score-processes', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def build_manifest(args) -> Path:
    with DESCRIPTIONS.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if '_take-1_' in row['scenario']]
    rows.sort(key=lambda row: int(row['scenario'].split('_', 1)[0]))
    if args.limit:
        rows = rows[:args.limit]

    switch_frames = args.benchmark_base / 'physics-IQ-benchmark-verified/switch-frames'
    output = args.output_root.resolve()
    cases = []
    for row in rows:
        file_id, view, _, scenario = row['scenario'].split('_', 3)
        name = row['generated_video_name']
        stem = Path(name).stem
        cases.append({
            'prompt': row['description'].strip(),
            'source_image': repo_relative(
                switch_frames / f'{file_id}_switch-frames_anyFPS_{view}_{Path(scenario).stem}.jpg'
            ),
            'image': repo_relative(
                output / 'prepared_images' / args.model_name / f'{stem}.jpg'
            ),
            'tokens': repo_relative(
                output / 'physical_language' / args.model_name / f'{stem}.json'
            ),
            'video': repo_relative(output / 'generated' / args.model_name / name),
        })

    manifest = output / 'manifests' / f'{args.model_name}.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(cases, indent=2) + '\n')
    print(f'Prepared {len(cases)} Physics-IQ cases')
    return manifest


def launch(stage, python, script, manifest, arguments, args):
    gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    log_dir = args.output_root / 'logs' / args.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for rank, gpu in enumerate(gpus):
        command = [
            python,
            str(script),
            '--manifest', str(manifest),
            '--rank', str(rank),
            '--world-size', str(len(gpus)),
            *arguments,
        ]
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu,
            IMAGE_MAX_TOKEN_NUM='1024',
            VIDEO_MAX_TOKEN_NUM='128',
            FPS_MAX_FRAMES='12',
            DIFFSYNTH_ATTENTION_IMPLEMENTATION='flash_attention_2',
            PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',
        )
        log_path = log_dir / f'{stage}-rank-{rank:02d}.log'
        log = log_path.open('w')
        process = subprocess.Popen(
            command,
            cwd=script.parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers.append((process, log, rank, log_path))
        print(f'[{stage}] rank={rank} gpu={gpu}')

    failures = []
    for process, log, rank, log_path in workers:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append(f'rank {rank}: {log_path}')
    if failures:
        raise RuntimeError(f'{stage} failed: ' + ', '.join(failures))


def run_reason(args, manifest):
    arguments = ['--model', args.reasoner_checkpoint]
    if args.overwrite:
        arguments.append('--overwrite')
    if args.no_progress:
        arguments.append('--no-progress')
    launch('reason', args.reasoner_python, REASONER, manifest, arguments, args)


def run_decode(args, manifest):
    arguments = ['--checkpoint', args.decoder_checkpoint]
    if args.dit_path:
        arguments.extend(('--dit-path', *args.dit_path))
    if args.vae_path:
        arguments.extend(('--vae-path', args.vae_path))
    if args.overwrite:
        arguments.append('--overwrite')
    if args.no_progress:
        arguments.append('--no-progress')
    launch('decode', args.decoder_python, DECODER, manifest, arguments, args)


def run_score(args):
    generated = args.output_root / 'generated' / args.model_name
    score_root = args.output_root / 'scores'
    command = [
        args.score_python or args.reasoner_python,
        str(args.benchmark_base / 'physiq/run_physics_iq.py'),
        '--input_folders', str(generated),
        '--output_folder', str(score_root),
        '--descriptions_file', str(DESCRIPTIONS),
        '--benchmark_base_folder', str(args.benchmark_base),
        '--n_process', str(args.score_processes),
    ]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.benchmark_base)
    environment['MPLCONFIGDIR'] = str(score_root / '.matplotlib')
    Path(environment['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=args.benchmark_base, env=environment, check=True)
    metrics = json.loads((
        score_root
        / 'physics-IQ-benchmark-verified/results'
        / f'{args.model_name}_metrics.json'
    ).read_text())
    print(f"Physics-IQ Verified: {float(metrics['final_score_view']) * 100:.2f}")


def main():
    args = parse_args()
    args.benchmark_base = args.benchmark_base.resolve()
    args.output_root = args.output_root.resolve()
    try:
        manifest = build_manifest(args)
        if args.mode in ('reason', 'generate', 'all'):
            run_reason(args, manifest)
        if args.mode in ('decode', 'generate', 'all'):
            run_decode(args, manifest)
        if args.mode in ('score', 'all'):
            run_score(args)
    finally:
        sanitize_outputs(args.output_root)


if __name__ == '__main__':
    main()
