<div align="center">

<img src="https://phi-zero.github.io/assets/img/phizero/logo.png" width="92" alt="PhiZero logo">

# PhiZero

### A World Model Built Around Physical Language

Shuyao Shang · Yuqi Wang · Ruopeng Gao · Xu Chen · Tieniu Tan · Lue Fan · Zhaoxiang Zhang

[![Project Page](https://img.shields.io/badge/Project-Page-1f63a3?style=for-the-badge)](https://phi-zero.github.io/)
[![Paper](https://img.shields.io/badge/Paper-PDF-d85b42?style=for-the-badge)](https://phi-zero.github.io/PhiZero.pdf)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-6f7780?style=for-the-badge)](#release-plan)
[![License](https://img.shields.io/badge/License-TBD-lightgrey?style=for-the-badge)](#license)

> **The code, pretrained models, and evaluation tools are being prepared for release.**
> Watch this repository to receive future updates.

</div>

<p align="center">
  <img src="https://phi-zero.github.io/assets/img/phizero/motivation.png" width="100%" alt="PhiZero motivation">
</p>

## Overview

**PhiZero** is a world model built around **physical language**: a compact, discrete
representation of world-state transitions learned self-supervisedly from unlabeled
in-the-wild videos.

Instead of directly predicting future pixels, PhiZero follows a
**reason-then-render** paradigm:

1. **Reason in physical language.** An autoregressive vision-language model predicts
   how the current world state will evolve.
2. **Render into video.** A video diffusion model translates the predicted transition
   sequence into the future visual observation.

This explicit transition space provides a reusable interface for physically realistic
video generation, interactive world exploration, action-conditioned simulation, and
zero-shot transfer across appearances and embodiments.

## Highlights

- **Physical language from video** — learns a discrete vocabulary of state transitions
  directly from large-scale, unlabeled, in-the-wild videos.
- **Structured world evolution** — separates reasoning about dynamics from rendering
  visual appearance.
- **Compact transition representation** — represents a four-second video with only
  256 physical-language tokens.
- **General-purpose world model** — supports physical video generation, driving and
  robotics control, cross-embodiment transfer, and sim-to-real transfer.
- **Strong physical reasoning** — achieves leading selected metrics on Physics-IQ
  Verified, PhyGround, WorldModelBench, and IntPhys2.

## Method

<p align="center">
  <img src="https://phi-zero.github.io/assets/img/phizero/Pipeline.webp" width="96%" alt="PhiZero method">
</p>

PhiZero contains two main components:

- The **Physical Language Tokenizer** uses a transition-level Q-Former and Finite
  Scalar Quantization to encode adjacent latent video states into discrete symbols.
- The **Physical Language Reasoner**, initialized from Qwen3-VL-4B, autoregressively
  predicts those symbols from a visual state and textual or numerical action intent.

A trained diffusion decoder then renders the inferred state transition into video.

## Results

| Benchmark | Selected metric | PhiZero |
|:--|:--|--:|
| Physics-IQ Verified | IQ-Score ↑ | **41.2** |
| PhyGround | Physics Score ↑ | **3.01** |
| WorldModelBench | Physics Adherence ↑ | **4.88** |
| IntPhys2 | Overall ↑ | **56.34** |

More quantitative and qualitative results are available on the
[project page](https://phi-zero.github.io/).

## Release Plan

- [x] Project page
- [x] Paper
- [ ] Inference code
- [ ] Pretrained Physical Language Tokenizer
- [ ] Pretrained Physical Language Reasoner and video decoder
- [ ] Training and fine-tuning recipes
- [ ] Evaluation scripts

We are actively preparing the implementation and model checkpoints for public
release. Exact release dates will be announced in this repository.

## Citation

If you find PhiZero useful for your research, please consider citing:

```bibtex
@article{shang2026phizero,
  title   = {PhiZero: A World Model Built Around Physical Language},
  author  = {Shang, Shuyao and Wang, Yuqi and Gao, Ruopeng and Chen, Xu
             and Tan, Tieniu and Fan, Lue and Zhang, Zhaoxiang},
  year    = {2026},
  note    = {Preprint}
}
```

## License

The source-code and model licenses will be announced together with the public
release.

## Acknowledgements

PhiZero builds on the progress of open video generation and vision-language
modeling. We thank the authors and contributors of the open-source projects and
benchmarks that made this research possible.

