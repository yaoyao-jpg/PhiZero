#!/usr/bin/env python3
"""Train PhiZero by reconstructing videos listed in a CSV."""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffsynth.training.phizero import PhiZeroDiffusionTrainingModel, VideoPathDataset


ROOT = Path(__file__).resolve().parents[3]
PATH_ARGUMENTS = {'csv', 'init_checkpoint', 'output_dir', 'dit_path', 'vae_path'}


def repo_relative(value: str | Path) -> str:
    path = Path(value).expanduser().resolve()
    return Path(os.path.relpath(path, ROOT)).as_posix()


def serialize_argument(key: str, value):
    if key not in PATH_ARGUMENTS or value is None:
        return value
    if isinstance(value, list):
        return [repo_relative(item) for item in value]
    return repo_relative(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dit-path', nargs='+')
    parser.add_argument('--vae-path')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--num-frames', type=int, default=33)
    parser.add_argument('--motion-height', type=int, default=256)
    parser.add_argument('--motion-width', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=100_000)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--lr-start', type=float, default=1e-6)
    parser.add_argument('--min-learning-rate', type=float, default=1e-6)
    parser.add_argument('--warmup-steps', type=int, default=1_000)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.95)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--motion-dropout', type=float, default=0.1)
    parser.add_argument('--save-every', type=int, default=5_000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--train-tokenizer', action='store_true')
    parser.add_argument('--fsq-entropy-weight', type=float, default=0.0)
    return parser.parse_args()


def build_lr_scheduler(optimizer, args):
    start_ratio = args.lr_start / args.learning_rate
    minimum_ratio = args.min_learning_rate / args.learning_rate

    def schedule(step):
        if step < args.warmup_steps:
            return start_ratio + (1.0 - start_ratio) * step / args.warmup_steps
        if args.max_steps <= args.warmup_steps:
            return 1.0
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(accelerator, model, output_dir: Path, step: int):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict = accelerator.unwrap_model(model).export_state_dict()
        save_file(state_dict, output_dir / f'step-{step}.safetensors')
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
        ],
    )
    set_seed(args.seed, device_specific=True)

    dataset = VideoPathDataset(args.csv, args.num_frames, args.height, args.width)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = PhiZeroDiffusionTrainingModel.from_pretrained(
        args.init_checkpoint,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        device=accelerator.device,
        motion_height=args.motion_height,
        motion_width=args.motion_width,
        motion_dropout=args.motion_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        train_tokenizer=args.train_tokenizer,
        fsq_entropy_weight=args.fsq_entropy_weight,
    )
    trainable_parameters = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, args)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            args.init_checkpoint.parent / 'model_config.json',
            output_dir / 'model_config.json',
        )
        values = {key: serialize_argument(key, value) for key, value in vars(args).items()}
        (output_dir / 'training_args.json').write_text(json.dumps(values, indent=2) + '\n')

    progress = tqdm(
        total=args.max_steps,
        disable=not accelerator.is_main_process,
        desc='Training',
    )
    global_step = 0
    last_saved_step = 0
    while global_step < args.max_steps:
        for video in dataloader:
            with accelerator.accumulate(model):
                loss = model(video)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            global_step += 1
            mean_loss = accelerator.gather(loss.detach()).mean().item()
            progress.update(1)
            progress.set_postfix(loss=f'{mean_loss:.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')

            if global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, output_dir, global_step)
                last_saved_step = global_step
            if global_step == args.max_steps:
                break

    if last_saved_step != global_step:
        save_checkpoint(accelerator, model, output_dir, global_step)
    progress.close()


if __name__ == '__main__':
    main()
