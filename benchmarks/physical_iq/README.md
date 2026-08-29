# PhiZero on Physics-IQ Verified

This directory generates and scores the 198-case PhiZero result on Physics-IQ
Verified. Reasoning and rendering run in separate environments, and each model
is loaded once per GPU shard.

## Benchmark data

Point `PHYSICS_IQ_BASE` at an official Physics-IQ checkout with this layout:

```text
<PHYSICS_IQ_BASE>/
├── physiq/run_physics_iq.py
└── physics-IQ-benchmark-verified/
    ├── switch-frames/
    ├── split-videos/
    └── video-masks/
```

The Verified dataset is gated. Request access on its
[Hugging Face page](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified)
and authenticate before downloading it. The following selective download is
enough for the complete 198-video inference and official scoring run:

```bash
git clone https://github.com/google-deepmind/physics-IQ-benchmark.git
cd physics-IQ-benchmark
uv sync

hf auth login
hf download Anates-Labs-Research/Physics-IQ-Verified \
  --repo-type dataset \
  --include 'switch-frames/*' \
  --include 'split-videos/testing/8FPS/*' \
  --include 'split-videos/testing/30FPS/*' \
  --include 'video-masks/real/8FPS/*' \
  --local-dir physics-IQ-benchmark-verified
```

The selected files total roughly 3.5 GB. `switch-frames` provides the 198
`take-1` inference inputs; the 8 FPS testing videos and real masks provide the
two physical takes used for normalization. The upstream scorer also validates
`split-videos/testing/30FPS` before using the 8 FPS data, so the 30 FPS source
videos are required even when every generated video is already 8 FPS. A
complete selective download contains 396 files in each of these four
directories.

Use that checkout's `.venv/bin/python` as `PHIZERO_SCORE_PYTHON`. This directory
includes the exact 396-row caption CSV used for the reported result; generation
uses its 198 `take-1` rows, while the official scorer uses both takes for
normalization.

## Evaluation settings

| Setting | Value |
| --- | --- |
| Input | 198 `take-1` switch frames, resized directly to 896x512 |
| Reasoner | greedy constrained decoding, 256 symbols, seed 1 |
| Decoder | Wan2.2-TI2V-5B, 50 steps, shift 5 |
| Native output | 33 frames at 8 FPS |
| Scoring output | frames 0-31 plus frame 32 repeated 8 times |
| Encoding | H.264, yuv420p, CRF 23, 40 frames at 8 FPS |
| Parallelism | 8 GPU workers per model stage |

The full run completed 198 token files, 198 videos, and all 66 scoring
scenarios. It reproduced **41.1 Physics-IQ Verified**, matching
[`reference_result.json`](reference_result.json).

## Run

First install the two model environments from the repository
[README](../../README.md#installation). Then set:

```bash
export PHIZERO_REASONER_PYTHON=/path/to/miniconda3/envs/phizero-reasoner/bin/python
export PHIZERO_DECODER_PYTHON=/path/to/miniconda3/envs/phizero-decoder/bin/python
export PHIZERO_SCORE_PYTHON=/path/to/physics-iq-benchmark/.venv/bin/python
export PHIZERO_REASONER_CHECKPOINT="${PWD}/ckpt/reasoner"
export PHIZERO_DECODER_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"
export PHYSICS_IQ_BASE=/path/to/physics-iq-benchmark
export PHIZERO_GPUS=0,1,2,3,4,5,6,7

bash benchmarks/physical_iq/run_physical_iq.sh
```

For local Wan weights:

```bash
export PHIZERO_DIT_PATHS="/path/to/dit-00001.safetensors /path/to/dit-00002.safetensors /path/to/dit-00003.safetensors"
export PHIZERO_VAE_PATH=/path/to/Wan2.2_VAE.safetensors
```

The default `all` mode builds the case manifest, runs reasoning, runs decoding,
and invokes the official scorer. Existing token files and videos are skipped.
Use `--overwrite` to regenerate them.

Common variants:

```bash
# One-GPU, one-case generation smoke test. Partial suites cannot be scored.
PHIZERO_MODE=generate PHIZERO_GPUS=0 \
  bash benchmarks/physical_iq/run_physical_iq.sh --limit 1 --no-progress

# Score an existing complete 198-video run.
PHIZERO_MODE=score bash benchmarks/physical_iq/run_physical_iq.sh
```

Outputs are kept separate from benchmark ground truth:

```text
outputs/
├── manifests/<model>.json
├── prepared_images/<model>/*.jpg
├── physical_language/<model>/*.json
├── generated/<model>/*.mp4
├── logs/<model>/{reason,decode}-rank-*.log
└── scores/physics-IQ-benchmark-verified/results/
```

The scorer defaults to one process because it can consume substantial CPU
memory. Set `PHIZERO_SCORE_PROCESSES` to increase its worker count.

## Licensing

Physics-IQ software is Apache-2.0 licensed and its other materials are CC-BY
4.0. The upstream notice is retained in `LICENSE.physics-iq`; the benchmark
videos and masks are not redistributed here.
