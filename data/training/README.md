# Training data

The data flow is: tokenizer videos -> tokenizer-generated Physical Language GT
-> reasoner SFT JSONL -> source-domain adaptation videos.

## 1. Physical Language Tokenizer videos

Tokenizer reconstruction training uses a CSV with one required column:

```csv
video
relative/or/absolute/path/to/clip.mp4
```

Paths are resolved relative to the CSV file. Each clip should cover 4 seconds
at 8 FPS. The loader takes the first 33 frames and deterministically resizes
them to 896x512. It performs no random crop, temporal sampling, color
transforms, or other data augmentation.

Point `PHIZERO_TRAIN_CSV` to this file and run
`examples/run_training.sh --train-tokenizer`. The release trainer starts from
`PHIZERO_INIT_CHECKPOINT`; it is a minimal fine-tuning path rather than a
from-scratch tokenizer pretraining recipe.

## 2. Physical Language ground truth

After tokenizer training, add a caption for every video:

```csv
video,prompt
relative/or/absolute/path/to/clip.mp4,A ball falls and bounces on the floor.
```

Set `PHIZERO_GT_CSV` to this file,
`PHIZERO_TOKENIZER_CHECKPOINT` to the trained tokenizer checkpoint, and run
`examples/run_physical_language_gt.sh`. The tokenizer-only exporter saves one
first frame per row and produces `train.jsonl` in the exact reasoner SFT format.
It loads neither the Wan DiT nor the Wan VAE.

## 3. Physical Language Reasoner SFT

`examples/run_reasoner_training.sh` accepts the generated JSONL through
`PHIZERO_REASONER_TRAIN_JSONL` and an optional validation JSONL through
`PHIZERO_REASONER_VAL_JSONL`. Each line has this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nText caption: {caption}\nPredict the motion token sequence."
    },
    {
      "role": "assistant",
      "content": "<bom><m_...><m_...>...<eom>"
    }
  ],
  "images": ["outputs/physical_language_gt/first_frames/00000000.jpg"]
}
```

The ellipsis is documentation only. A real assistant target contains exactly
256 adjacent symbols in the range `<m_0>` through `<m_24999>`. Reasoner SFT
reads only the first frame, caption, and discrete target sequence; it does not
run or backpropagate through the tokenizer.

The release does not redistribute the large-scale reasoner training set. The
launcher validates every supplied JSONL row and creates the required 25,002
atomic special tokens before starting ms-swift.

## 4. Source-domain adaptation

[`toy_videos.csv`](toy_videos.csv) contains exactly the 11 source clips from the
motion-transfer examples. `examples/run_motion_transfer_adaptation.sh` uses it
by default and updates the Physical Language Tokenizer, conditioner, and DiT
LoRA through source-video reconstruction. Captions, target videos, and paired
source-target data are not required.

For a different source domain, point `PHIZERO_ADAPT_CSV` to another one-column
`video` CSV with the same format as the tokenizer training list.
