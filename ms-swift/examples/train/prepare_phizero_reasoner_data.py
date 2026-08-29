#!/usr/bin/env python3
"""Validate PhiZero reasoner SFT JSONL and write its atomic token vocabulary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CODEBOOK_SIZE = 25_000
PHYSICAL_LANGUAGE_LENGTH = 256
PROMPT_PREFIX = "<image>\nText caption: "
PROMPT_SUFFIX = "\nPredict the motion token sequence."
BEGIN_TOKEN = "<bom>"
END_TOKEN = "<eom>"
SYMBOL_RE = re.compile(r"<m_([0-9]+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--tokens-output", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def fail(path: Path, line_number: int, message: str) -> None:
    raise ValueError(f"{path}:{line_number}: {message}")


def validate_messages(path: Path, line_number: int, row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        fail(
            path,
            line_number,
            "messages must contain exactly one user turn and one assistant turn",
        )

    user, assistant = messages
    if not isinstance(user, dict) or user.get("role") != "user":
        fail(path, line_number, "the first message must have role=user")
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        fail(path, line_number, "the second message must have role=assistant")

    user_content = user.get("content")
    if not isinstance(user_content, str):
        fail(path, line_number, "user content must be a string")
    if not user_content.startswith(PROMPT_PREFIX) or not user_content.endswith(
        PROMPT_SUFFIX
    ):
        fail(path, line_number, "user content does not use the PhiZero reasoner prompt")
    caption = user_content[len(PROMPT_PREFIX) : -len(PROMPT_SUFFIX)]
    if not caption.strip():
        fail(path, line_number, "caption is empty")

    target = assistant.get("content")
    if (
        not isinstance(target, str)
        or not target.startswith(BEGIN_TOKEN)
        or not target.endswith(END_TOKEN)
    ):
        fail(path, line_number, "assistant target must be <bom><m_i>...<eom>")
    body = target[len(BEGIN_TOKEN) : -len(END_TOKEN)]
    matches = list(SYMBOL_RE.finditer(body))
    if "".join(match.group(0) for match in matches) != body:
        fail(
            path,
            line_number,
            "assistant target contains text outside contiguous <m_i> symbols",
        )
    indices = [int(match.group(1)) for match in matches]
    if "".join(f"<m_{index}>" for index in indices) != body:
        fail(
            path,
            line_number,
            "assistant target contains a non-canonical symbol such as <m_01>",
        )
    if len(indices) != PHYSICAL_LANGUAGE_LENGTH:
        fail(
            path,
            line_number,
            f"assistant target has {len(indices)} symbols; expected {PHYSICAL_LANGUAGE_LENGTH}",
        )
    invalid = next((index for index in indices if not 0 <= index < CODEBOOK_SIZE), None)
    if invalid is not None:
        fail(
            path,
            line_number,
            f"physical-language index {invalid} is outside [0, {CODEBOOK_SIZE})",
        )


def validate_images(
    path: Path,
    line_number: int,
    row: dict[str, Any],
    check_images: bool,
    path_root: Path,
) -> None:
    images = row.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        fail(path, line_number, "images must contain exactly one image path")
    image = Path(images[0])
    if not image.is_absolute():
        image = path_root / image
    if check_images and not image.is_file():
        fail(path, line_number, f"image does not exist: {image}")


def validate_dataset(path: Path, check_images: bool, path_root: Path) -> int:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                fail(path, line_number, "blank lines are not allowed")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(path, line_number, f"invalid JSON: {error.msg}")
            if not isinstance(row, dict):
                fail(path, line_number, "each JSONL row must be an object")
            validate_messages(path, line_number, row)
            validate_images(path, line_number, row, check_images, path_root)
            row_count += 1

    if row_count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    print(f"Validated {row_count} reasoner samples: {path}")
    return row_count


def write_special_tokens(path: Path) -> None:
    expected = "".join(
        [
            f"{BEGIN_TOKEN}\n",
            f"{END_TOKEN}\n",
            *(f"<m_{index}>\n" for index in range(CODEBOOK_SIZE)),
        ]
    )
    if path.is_file() and path.read_text(encoding="utf-8") == expected:
        print(f"Physical-language vocabulary is already valid: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    print(f"Wrote {CODEBOOK_SIZE + 2} atomic special tokens: {path}")


def main() -> None:
    args = parse_args()
    path_root = args.path_root.resolve()
    seen: set[Path] = set()
    for dataset in args.dataset:
        dataset = dataset.resolve()
        if dataset not in seen:
            validate_dataset(
                dataset,
                check_images=not args.skip_image_check,
                path_root=path_root,
            )
            seen.add(dataset)
    write_special_tokens(args.tokens_output.resolve())


if __name__ == "__main__":
    main()
