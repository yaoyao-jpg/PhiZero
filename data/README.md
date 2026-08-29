# PhiZero release data

`inference_cases/` contains the 19 public examples used by
`examples/run_inference_cases.sh`. `cases.json` stores only the case ID, relative
first-frame path, and prompt. Generated tokens and videos are written under
`outputs/` rather than stored with the inputs.

`motion_transfer/` contains 11 source-video/target-first-frame pairs for
`examples/run_motion_transfer.sh`.

`reconstruction/` contains six source videos for
`examples/run_reconstruction.sh`.

`training/` documents two independent training formats. The included
one-column video CSV is consumed by the decoder-refinement and
motion-transfer-adaptation launchers. Physical Language Reasoner SFT instead
uses user-provided multimodal JSONL as described in `training/README.md`.
