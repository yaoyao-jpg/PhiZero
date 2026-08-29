# Public inference cases

This directory contains all 19 paper examples. The images are the clean
896x512 conditioning frames, without the caption panel used in the paper
visualization.

`cases.json` is a plain list of:

```json
{
  "id": "case_infer_03",
  "image": "first_frames/case_infer_03.jpg",
  "prompt": "A pre-sliced whole orange bursts apart in slow motion ..."
}
```

Run one example:

```bash
bash examples/run_inference_cases.sh --case case_infer_03
```

Omit `--case` to run all 19. Outputs are kept equally small:

```text
outputs/inference_cases/
├── physical_language/<case-id>.json  # 256 integer indices
└── videos/<case-id>.mp4
```

Existing files are skipped; pass `--overwrite` to regenerate them. The image
assets retain any applicable upstream dataset terms.

The runner only accepts case IDs from this checked-in file; it has no external
image, prompt, manifest, or output-directory interface.
