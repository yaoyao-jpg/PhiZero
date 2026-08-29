<div align="center">

<p>
  <img src="https://phi-zero.github.io/assets/img/phizero/logo.png" height="76" alt="PhiZero logo">
  &nbsp;&nbsp;&nbsp;
  <img src="https://phi-zero.github.io/assets/img/phizero/wordmark.png" height="76" alt="PhiZero">
</p>

### A World Model Built Around Physical Language

Shuyao Shang · Yuqi Wang · Ruopeng Gao · Xu Chen · Tieniu Tan · Lue Fan · Zhaoxiang Zhang

[![Project Page](https://img.shields.io/badge/Project-Page-1f63a3?style=for-the-badge)](https://phi-zero.github.io/)
[![Paper](https://img.shields.io/badge/Paper-PDF-d85b42?style=for-the-badge)](https://arxiv.org/pdf/2607.28624)
[![Code](https://img.shields.io/badge/Code-Released-2ea44f?style=for-the-badge)](https://github.com/yaoyao-jpg/PhiZero)

</div>

PhiZero is a world model that reasons about future dynamics in a discrete
**Physical Language** before rendering that reasoning as video.

<div align="center">
  <video src="assets/PhiZero.mp4" width="100%" controls></video>
</div>


## Repository layout

```text
PhiZero/
├── ckpt/
│   ├── reasoner/                     # Physical Language Reasoner checkpoint
│   └── tokenizer/
│       ├── tokenizer.safetensors     # tokenizer, conditioner, and DiT LoRA
│       └── model_config.json         # checkpoint architecture
├── examples/run_inference_cases.sh    # launcher for the 19 paper examples
├── examples/run_motion_transfer.sh    # launcher for 11 transfer examples
├── examples/run_motion_transfer_adaptation.sh
│                                      # source-only reconstruction adaptation
├── examples/run_reconstruction.sh     # launcher for 6 reconstruction examples
├── examples/run_physical_language_gt.sh
│                                      # tokenizer-only reasoner GT extraction
├── examples/run_reasoner_training.sh  # Physical Language Reasoner full SFT
├── examples/run_training.sh           # video-only decoder training launcher
├── examples/PROMPT_GUIDANCE.txt        # training-aligned caption pattern
├── data/inference_cases/              # first frames, prompts, and cases.json
├── data/motion_transfer/              # source videos and target first frames
├── data/reconstruction/                # six source videos and cases.json
├── data/training/toy_videos.csv        # 11-video toy training list
├── benchmarks/physical_iq/            # Physics-IQ generation and scoring
├── environments/                      # reproducible Conda and pip specs
├── ms-swift/
│   ├── swift/infer_engine/physical_language.py
│   ├── examples/infer/phizero_reason*.py
│   └── examples/train/prepare_phizero_reasoner_data.py
└── DiffSynth-Studio/
    ├── diffsynth/models/phizero/       # physical-language tokenizer and inverse FSQ
    ├── diffsynth/pipelines/phizero.py
    ├── diffsynth/training/phizero.py
    └── examples/phizero/               # GT extraction, rendering, transfer, and training
```

## Checkpoints

The released checkpoints are hosted in
[ShuyaoShang/PhiZero](https://huggingface.co/ShuyaoShang/PhiZero) model
repository. Download them from the PhiZero repository root:

```bash
hf download ShuyaoShang/PhiZero --local-dir ckpt
```

This creates the checkpoint layout expected by all training and inference
entry points:

```text
ckpt/reasoner
ckpt/tokenizer/tokenizer.safetensors
```

Keep `ckpt/tokenizer/model_config.json` beside `tokenizer.safetensors`. The
decoder constructs the tokenizer, QFormer, FSQ, connector, and projector from
this configuration. The tokenizer checkpoint also contains the inverse-FSQ
context tail and DiT LoRA used for rendering. The official Wan base DiT and VAE
are resolved separately by DiffSynth or through `PHIZERO_DIT_PATHS` and
`PHIZERO_VAE_PATH`.


## Installation

The checked-in environment files reproduce the Linux x86-64, CUDA 12.4 stack
used for the released result. Run the commands from the repository root and
point `PHIZERO_CONDA` at the desired Miniconda installation:

```bash
PHIZERO_CONDA=/path/to/miniconda3/bin/conda
```

Create the reasoner environment:

```bash
CONDARC=environments/condarc "${PHIZERO_CONDA}" env create \
  -f environments/phizero-reasoner.yml
"${PHIZERO_CONDA}" run -n phizero-reasoner \
  python -m pip install -r environments/requirements-reasoner.txt
PIP_CACHE_DIR=/tmp/phizero-pip-cache CUDA_HOME=/usr/local/cuda MAX_JOBS=8 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 \
  "${PHIZERO_CONDA}" run -n phizero-reasoner \
  python -m pip install --no-deps --no-build-isolation --no-binary flash-attn \
  "flash-attn==2.8.3"
"${PHIZERO_CONDA}" run -n phizero-reasoner \
  python -m pip install --no-deps --no-build-isolation -e ./ms-swift
```

Create the decoder environment:

```bash
CONDARC=environments/condarc "${PHIZERO_CONDA}" env create \
  -f environments/phizero-decoder.yml
"${PHIZERO_CONDA}" run -n phizero-decoder \
  python -m pip install -r environments/requirements-decoder.txt
PIP_CACHE_DIR=/tmp/phizero-pip-cache CUDA_HOME=/usr/local/cuda MAX_JOBS=8 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 \
  "${PHIZERO_CONDA}" run -n phizero-decoder \
  python -m pip install --no-deps --no-build-isolation --no-binary flash-attn \
  "flash-attn==2.8.3"
"${PHIZERO_CONDA}" run -n phizero-decoder \
  python -m pip install --no-deps --no-build-isolation -e ./DiffSynth-Studio
```

### Download the Wan2.2 base weights

The PhiZero tokenizer checkpoint contains the Physical Language modules and
the rendering LoRA, but not the frozen Wan2.2 base model. Download the three
official Wan2.2-TI2V-5B DiT shards and the DiffSynth-converted VAE from the
repository root. The decoder environment already installs both the `hf` and
`modelscope` CLIs used below:

```bash
mkdir -p ckpt/wan2.2-ti2v-5b ckpt/wan2.2-vae

hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model-*.safetensors' \
  --local-dir ckpt/wan2.2-ti2v-5b

modelscope download \
  --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  Wan2.2_VAE.safetensors \
  --local_dir ckpt/wan2.2-vae
```

Point inference, training, and benchmark launchers at these local files:

```bash
export PHIZERO_DIT_PATHS="${PWD}/ckpt/wan2.2-ti2v-5b/diffusion_pytorch_model-00001-of-00003.safetensors ${PWD}/ckpt/wan2.2-ti2v-5b/diffusion_pytorch_model-00002-of-00003.safetensors ${PWD}/ckpt/wan2.2-ti2v-5b/diffusion_pytorch_model-00003-of-00003.safetensors"
export PHIZERO_VAE_PATH="${PWD}/ckpt/wan2.2-vae/Wan2.2_VAE.safetensors"
```

The DiT shards require about 20 GB and the VAE about 1.4 GB. If these
variables are unset, DiffSynth can resolve the same base models automatically;
explicit local paths make offline and repeated runs reproducible.

The same decoder environment contains all dependencies required by the
video-only trainer and Physical Language GT exporter.

`CUDA_HOME` must contain `bin/nvcc`. `FLASH_ATTN_CUDA_ARCHS=80` targets A100;
use `90` for H100. Building FlashAttention locally links it against the exact
PyTorch ABI installed in each environment.

Check both environments:

```bash
"${PHIZERO_CONDA}" run -n phizero-reasoner python -m pip check
"${PHIZERO_CONDA}" run -n phizero-reasoner python -c \
  "import deepspeed, flash_attn, torch, transformers; from swift.infer_engine import PhysicalLanguageReasoner; print(torch.__version__, transformers.__version__, flash_attn.__version__, deepspeed.__version__)"

DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 \
  "${PHIZERO_CONDA}" run -n phizero-decoder python -m pip check
DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 \
  "${PHIZERO_CONDA}" run -n phizero-decoder python -c \
  "import flash_attn, torch; from diffsynth.pipelines.phizero import PhiZeroDiffusionDecoder; print(torch.__version__, flash_attn.__version__)"
```

Both environments passed `pip check` and reproduced the released inference
path. The reasoner trainer pins the same DeepSpeed 0.18.9 version used by the
original full-SFT run.

## Training workflow

The release follows the dependency order below. Physical Language targets must
always be regenerated after changing the tokenizer used for a reasoner run.

1. Train or fine-tune the Physical Language Tokenizer on videos.
2. Freeze that tokenizer and extract Physical Language ground truth from
   captioned videos.
3. Train the Physical Language Reasoner on the exported first-frame/caption/GT
   JSONL.
4. Before cross-domain motion transfer, adapt the tokenizer/decoder on videos
   from the source-motion domain.
5. Run reason-then-render inference, reconstruction, or motion transfer.

## 1. Train the Physical Language Tokenizer

Tokenizer training metadata is a CSV with one required column:

```csv
video
/path/to/clip_000.mp4
relative/path/to/clip_001.mp4
```

Relative paths are resolved from the CSV directory. Each path should be a
4-second, 8 FPS clip. The loader takes the first 33 frames and deterministically
resizes them to 896x512; it does not apply random crop, temporal sampling,
color transforms, Cosmos augmentation, or other data augmentation.

The minimal release trainer starts from a compatible PhiZero checkpoint and
uses video reconstruction to jointly optimize the Physical Language Tokenizer,
inverse-FSQ/context conditioner, and DiT LoRA. The Wan base DiT and VAE remain
frozen:

```bash
export PHIZERO_TRAIN_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_INIT_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"
export PHIZERO_TRAIN_CSV=/path/to/tokenizer_videos.csv
export PHIZERO_TRAIN_OUTPUT_DIR="${PWD}/outputs/tokenizer_training"
export PHIZERO_MAX_STEPS=100000
export PHIZERO_DIT_PATHS="/path/to/dit-00001.safetensors /path/to/dit-00002.safetensors /path/to/dit-00003.safetensors"
export PHIZERO_VAE_PATH=/path/to/Wan2.2_VAE.safetensors

bash examples/run_training.sh \
  --train-tokenizer \
  --fsq-entropy-weight 0.05 \
  --gradient-checkpointing
```

Set `PHIZERO_NUM_PROCESSES` for single-node DDP. Extra arguments are forwarded
to `train.py`. Each save writes `step-N.safetensors`, `model_config.json`, and
`training_args.json`; keep `model_config.json` beside the checkpoint. Omitting
`--train-tokenizer` freezes the tokenizer and performs decoder-only refinement.

The default optimization settings are batch size 1 per process, gradient
accumulation 2, AdamW `(0.9, 0.95)`, learning rate `2e-5`, 1,000 warmup steps,
cosine decay to `1e-6`, and 100,000 optimizer steps. Future-latent flow matching
is used for reconstruction; the clean first-frame latent is excluded from the
loss.

## 2. Generate Physical Language Ground Truth

After tokenizer training, prepare a CSV containing the same 33-frame videos and
their captions:

```csv
video,prompt
/path/to/clip_000.mp4,A ball falls and bounces on the floor.
relative/path/to/clip_001.mp4,A person pushes a box across the floor.
```

Then run the tokenizer-only exporter. It does not load the Wan DiT or VAE:

```bash
export PHIZERO_DECODER_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_TOKENIZER_CHECKPOINT="${PWD}/outputs/tokenizer_training/step-100000.safetensors"
export PHIZERO_GT_CSV=/path/to/reasoner_videos_and_prompts.csv
export PHIZERO_GT_OUTPUT_DIR="${PWD}/outputs/physical_language_gt"

bash examples/run_physical_language_gt.sh
```

The exporter takes the first 33 frames, saves the resized first frame under
`outputs/physical_language_gt/first_frames/`, and writes
`outputs/physical_language_gt/train.jsonl`. Every assistant target is exactly
`<bom>` + 256 adjacent `<m_i>` symbols + `<eom>`, with repository-root-relative
image paths and the fixed PhiZero user prompt. Use a separate CSV/output
directory for a validation split. Pass `--overwrite` only when intentionally
replacing an existing JSONL.

## 3. Train the Physical Language Reasoner

Reasoner training consumes the JSONL generated in the previous step. Each row
contains one first frame, the fixed task prompt, and exactly 256 ground-truth
Physical Language symbols:

```json
{"messages":[{"role":"user","content":"<image>\nText caption: A ball falls and bounces on the floor.\nPredict the motion token sequence."},{"role":"assistant","content":"<bom><m_10487><m_782>...<m_9231><eom>"}],"images":["outputs/physical_language_gt/first_frames/00000000.jpg"]}
```

The ellipsis above is illustrative; real rows contain 256 contiguous symbols
with `0 <= i < 25000` and no spaces. The tokenizer and diffusion decoder do not
participate in reasoner backpropagation.

Run full-parameter Qwen3-VL SFT in the reasoner environment:

```bash
export PHIZERO_REASONER_PYTHON=/path/to/phizero-reasoner/bin/python
export PHIZERO_REASONER_TRAIN_JSONL="${PWD}/outputs/physical_language_gt/train.jsonl"
# Optional, but recommended:
export PHIZERO_REASONER_VAL_JSONL=/path/to/validation.jsonl

# Defaults to ckpt/reasoner, one GPU, one epoch, and outputs/reasoner_training.
bash examples/run_reasoner_training.sh

# Single-node multi-GPU; ZeRO-2 is enabled automatically.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PHIZERO_REASONER_NUM_PROCESSES=8 \
bash examples/run_reasoner_training.sh
```

Set `PHIZERO_REASONER_INIT_MODEL` to a Qwen3-VL-4B base model or a compatible
earlier reasoner checkpoint. Before launch, the wrapper validates every JSONL
row and creates the exact 25,002-token atomic vocabulary (`<bom>`, `<eom>`, and
`<m_0>` through `<m_24999>`). Set `PHIZERO_REASONER_SKIP_IMAGE_CHECK=1` only
when image files are not visible on the launch host. Other ms-swift arguments
can be appended to the command.

The released model used two successive runs of this same objective: a general
motion-token stage followed by SFT on physics-filtered data. Both stages train
the language model, visual encoder, and aligner with autoregressive
cross-entropy on the assistant sequence only; there is no diffusion or pixel
loss. The final released SFT used 8 nodes x 8 GPUs, batch size 8 per GPU,
gradient accumulation 1 (global batch 512), 3 epochs, and ZeRO-2. The minimal
launcher retains learning rate `1e-5`, cosine decay, 5% warmup, weight decay
`0.1`, Adam beta2 `0.95`, bfloat16 FlashAttention, gradient checkpointing, and
sequence length 1024, while defaulting to one GPU, batch size 1, and one epoch.

## 4. Source-domain adaptation

Before motion transfer, adapt on videos from the domain that supplies the
motion: for example, human-hand videos for human-to-dexterous-hand transfer or
simulated robot videos for sim-to-real transfer. If no broader domain-matched
collection is available, using each source video itself is sufficient. Target
videos and paired source-target data are not used.

The included [`data/training/toy_videos.csv`](data/training/toy_videos.csv)
contains exactly the 11 released source videos. This command directly runs the
source-only reconstruction adaptation for 1,000 optimizer steps:

```bash
export PHIZERO_TRAIN_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_INIT_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"
export PHIZERO_DIT_PATHS="/path/to/dit-00001.safetensors /path/to/dit-00002.safetensors /path/to/dit-00003.safetensors"
export PHIZERO_VAE_PATH=/path/to/Wan2.2_VAE.safetensors

bash examples/run_motion_transfer_adaptation.sh
```

This jointly adapts the tokenizer, conditioner, and DiT LoRA and writes
`outputs/motion_transfer_adaptation/step-1000.safetensors`. For another domain,
set `PHIZERO_ADAPT_CSV` to a one-column `video` CSV. The variables
`PHIZERO_ADAPT_STEPS`, `PHIZERO_ADAPT_OUTPUT_DIR`, and
`PHIZERO_NUM_PROCESSES` control the run.

## Inference examples

[`data/inference_cases/cases.json`](data/inference_cases/cases.json) is a plain
list of 19 entries. Each entry contains only an ID, a relative first-frame path,
and a fixed prompt:

```json
{
  "id": "case_infer_03",
  "image": "first_frames/case_infer_03.jpg",
  "prompt": "A pre-sliced whole orange bursts apart in slow motion ..."
}
```

Set the two environment paths, point to the included checkpoints, and run one
or all cases from the repository root:

```bash
export PHIZERO_REASONER_PYTHON=/path/to/phizero-reasoner/bin/python
export PHIZERO_DECODER_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_REASONER_CHECKPOINT="${PWD}/ckpt/reasoner"
export PHIZERO_DECODER_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"

# One released case
bash examples/run_inference_cases.sh --case case_infer_03

# All 19 cases
bash examples/run_inference_cases.sh
```

The release inference entry point only reads this checked-in manifest and its
19 images and captions. It does not accept an external image, prompt, manifest,
or output directory. The caption style used for these examples is documented
in [`examples/PROMPT_GUIDANCE.txt`](examples/PROMPT_GUIDANCE.txt).

Optional local Wan paths can be set with `PHIZERO_DIT_PATHS` and
`PHIZERO_VAE_PATH`. Outputs go to `outputs/inference_cases/`; existing token
files and videos are skipped unless `--overwrite` is passed. The 19-case run
was verified with the released checkpoints: every token file contains 256
indices and every video is 896x512, 33 frames, and 8 FPS.

## Motion transfer

> [!IMPORTANT]
> The included `ckpt/tokenizer/tokenizer.safetensors` is a general initialization
> checkpoint and cannot be used directly for robot-domain or sim-to-real motion
> transfer. First fine-tune it with videos from the same domain as the source
> motion. If no additional domain-matched data are available, simply train on
> the source videos themselves through reconstruction; target videos and paired
> source-target data are not required.

```bash
# Run from the repository root; this trains on all 11 included source videos.
export PHIZERO_TRAIN_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_INIT_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"
export PHIZERO_DIT_PATHS="/path/to/dit-00001.safetensors /path/to/dit-00002.safetensors /path/to/dit-00003.safetensors"
export PHIZERO_VAE_PATH=/path/to/Wan2.2_VAE.safetensors
bash examples/run_motion_transfer_adaptation.sh
```

The included transfer set has 3 ego-to-dexterous-robot, 4 human-to-robot, and
4 simulation-to-real cases. Each entry in
[`data/motion_transfer/cases.json`](data/motion_transfer/cases.json) contains
only an ID, a source-video path, and a target-first-frame path:

```json
{
  "id": "human2robot_transfer_01",
  "source_video": "cases/human2robot/human2robot_transfer_01/source_video.mp4",
  "target_image": "cases/human2robot/human2robot_transfer_01/human2robot_transfer_01.png"
}
```

Then run motion transfer with the adapted checkpoint:

```bash
export PHIZERO_DECODER_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_DECODER_CHECKPOINT="${PWD}/outputs/motion_transfer_adaptation/step-1000.safetensors"

# Inspect the case IDs without loading a model
bash examples/run_motion_transfer.sh --list-cases

# One case
bash examples/run_motion_transfer.sh --case human2robot_transfer_01

# All 11 cases
bash examples/run_motion_transfer.sh
```

Local Wan weights use the same `PHIZERO_DIT_PATHS` and `PHIZERO_VAE_PATH`
variables as above. Outputs are raw 256-symbol arrays under
`outputs/motion_transfer/physical_language/` and 33-frame videos under
`outputs/motion_transfer/videos/`.

## Reconstruction

[`data/reconstruction/cases.json`](data/reconstruction/cases.json) lists six
33-frame source videos using the minimal `id`/`video` format:

```json
{
  "id": "recon_01",
  "video": "videos/recon_01.mp4"
}
```

The tokenizer extracts Physical Language from the source video, and the
decoder renders it from the source video's own first frame:

```bash
export PHIZERO_DECODER_PYTHON=/path/to/phizero-decoder/bin/python
export PHIZERO_DECODER_CHECKPOINT="${PWD}/ckpt/tokenizer/tokenizer.safetensors"

# Inspect case IDs without loading a model
bash examples/run_reconstruction.sh --list-cases

# One case
bash examples/run_reconstruction.sh --case recon_01

# All six cases
bash examples/run_reconstruction.sh
```

Local Wan paths use `PHIZERO_DIT_PATHS` and `PHIZERO_VAE_PATH`. Outputs are raw
symbols under `outputs/reconstruction/physical_language/` and reconstructed
videos under `outputs/reconstruction/videos/`. The included six-case run was
verified with the released checkpoint; every output contains 256 symbols and a
896x512, 33-frame, 8 FPS video.

## Physics-IQ Verified

After requesting access to the gated
[`Anates-Labs-Research/Physics-IQ-Verified`](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified)
dataset and running `hf auth login`, download only the data used by inference
and scoring:

```bash
export PHYSICS_IQ_BASE=/path/to/physics-IQ-benchmark

hf download Anates-Labs-Research/Physics-IQ-Verified \
  --repo-type dataset \
  --include 'switch-frames/*' \
  --include 'split-videos/testing/8FPS/*' \
  --include 'split-videos/testing/30FPS/*' \
  --include 'video-masks/real/8FPS/*' \
  --local-dir "${PHYSICS_IQ_BASE}/physics-IQ-benchmark-verified"
```

The upstream scorer validates the 30 FPS source videos even though PhiZero
outputs and final metric inputs are 8 FPS, so do not omit the `30FPS` pattern.
See [`benchmarks/physical_iq/README.md`](benchmarks/physical_iq/README.md) for
benchmark installation, environment variables, and launch commands.

## Contact Us

We believe PhiZero has the potential to unlock a wide range of applications,
and we are excited to see how the community will use and build upon it. If you
have any questions or suggestions, or would like to share your results, please
feel free to contact Shuyao Shang via email at
[shangshuyao2024@ia.ac.cn](mailto:shangshuyao2024@ia.ac.cn) or WeChat
(`17315710019`). We also welcome GitHub issues for bug reports and feature
requests.
