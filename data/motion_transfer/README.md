# Motion-transfer examples

The included general tokenizer checkpoint cannot be used directly for the
robot-domain or sim-to-real transfers. It must first be adapted on videos from
the same domain as the source motion. If no separate same-domain dataset is
available, the source videos themselves are sufficient: run
`examples/run_motion_transfer_adaptation.sh` to train by reconstructing the 11
included source videos. This requires neither target videos nor paired data.

Each case contains only a 33-frame, 8 FPS, 896x512 source video and an
896x512 target first frame. `cases.json` supplies the relative paths consumed
by `examples/run_motion_transfer.sh`.

The target frames are curated appearance edits whose geometry follows the
source first frame. Generated physical-language sequences and videos are
written under `outputs/motion_transfer/`.

Case IDs are grouped as `ego2dex_transfer_01`-`03`,
`human2robot_transfer_01`-`04`, and `sim2real_transfer_01`-`04`. The runner only
accepts these checked-in cases.
