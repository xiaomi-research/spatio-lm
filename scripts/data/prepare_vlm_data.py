#!/usr/bin/env python3
"""Convert and validate VSI-590K-style annotations for SpatioLM training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE = "<image>"
VIDEO = "<video>"
ROLES = {
    "human": "user",
    "user": "user",
    "question": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "answer": "assistant",
    "model": "assistant",
    "system": "system",
}


class RecordError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", dest="inputs")
    parser.add_argument("--output", help="Output JSONL file or directory.")
    parser.add_argument(
        "--media-root", help="Media root; defaults to the input directory."
    )
    parser.add_argument("--sample", type=int, help="Keep at most N valid records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def source_files(inputs: List[str]) -> List[Path]:
    files = []
    for value in inputs:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError("Input does not exist: {}".format(path))
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".ndjson"}
                )
            )
    result = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    if not result:
        raise ValueError("No JSON/JSONL files found.")
    return result


def records(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = next(
                (payload[key] for key in ("data", "items", "records", "annotations") if isinstance(payload.get(key), list)),
                [payload],
            )
        else:
            raise ValueError("Top-level JSON must be an object or list: {}".format(path))
        for index, value in enumerate(values, 1):
            if not isinstance(value, dict):
                raise RecordError("record is not an object")
            yield index, value
        return

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RecordError("record is not an object")
                yield line_number, value


def text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("text") is not None:
                    chunks.append(str(item["text"]))
                elif item.get("type") in {"image", "image_url"}:
                    chunks.append(IMAGE)
                elif item.get("type") == "video":
                    chunks.append(VIDEO)
        return "".join(chunks).strip()
    return "" if value is None else str(value).strip()


def media_list(value: Any) -> List[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
        raise RecordError("media paths must be strings")
    return [item.strip() for item in values]


def messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    values = record.get("messages") or record.get("conversations")
    if values is None and ("question" in record or "answer" in record):
        values = [
            {"role": "user", "content": record.get("question")},
            {"role": "assistant", "content": record.get("answer")},
        ]
    if not isinstance(values, list) or not values:
        raise RecordError("missing messages/conversations")
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise RecordError("message is not an object")
        role = ROLES.get(str(value.get("role", value.get("from", ""))).lower())
        content = text(value.get("content", value.get("value")))
        if role is None or not content:
            raise RecordError("message role or content is invalid")
        result.append({"role": role, "content": content})
    if not any(item["role"] == "user" for item in result):
        raise RecordError("conversation has no user message")
    if not any(item["role"] == "assistant" for item in result):
        raise RecordError("conversation has no assistant message")
    return result


def relative_path(value: str, source: Path, root: Optional[Path]) -> str:
    value = value[7:] if value.startswith("file://") else value
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    for base in (root, source.parent):
        if base is not None:
            try:
                return path.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                pass
    return str(path)


def normalize(record: Dict[str, Any], source: Path, root: Optional[Path]) -> Dict[str, Any]:
    result = {"messages": messages(record)}
    for key, singular in (("images", "image"), ("videos", "video")):
        values = media_list(record.get(key, record.get(singular)))
        if values:
            result[key] = [relative_path(value, source, root) for value in values]
        tokens = sum(item["content"].count(IMAGE if key == "images" else VIDEO) for item in result["messages"])
        if tokens != len(values):
            raise RecordError("{} placeholders ({}) != paths ({})".format(key, tokens, len(values)))
    if not result.get("images") and not result.get("videos"):
        raise RecordError("record has no images or videos")
    if "question_type" in record:
        result["question_type"] = record["question_type"]
    return result


def check_media(record: Dict[str, Any], source: Path, root: Optional[Path]) -> None:
    base = root or source.parent
    for key in ("images", "videos"):
        for value in record.get(key, []):
            path = Path(value)
            path = path if path.is_absolute() else base / path
            if not path.is_file():
                raise RecordError("media file does not exist: {}".format(path))


def main() -> int:
    args = parse_args()
    if args.sample is not None and args.sample < 0:
        raise ValueError("--sample must be non-negative")
    if not args.check_only and not args.output:
        raise ValueError("--output is required unless --check-only is used")
    files = source_files(args.inputs)
    root = Path(args.media_root).expanduser() if args.media_root else None
    if root is None and len(args.inputs) == 1:
        input_path = Path(args.inputs[0]).expanduser()
        root = input_path if input_path.is_dir() else input_path.parent

    valid, errors, total = [], [], 0
    for file in files:
        try:
            for line, record in records(file):
                total += 1
                try:
                    item = normalize(record, file, root)
                    check_media(item, file, root)
                    valid.append(item)
                except (RecordError, json.JSONDecodeError) as exc:
                    errors.append("{}:{}: {}".format(file, line, exc))
        except (OSError, ValueError, RecordError, json.JSONDecodeError) as exc:
            errors.append("{}: {}".format(file, exc))

    if args.deduplicate:
        unique = {}
        for item in valid:
            key = sha1(json.dumps(item, sort_keys=True).encode()).hexdigest()
            unique.setdefault(key, item)
        valid = list(unique.values())
    if args.sample is not None and args.sample < len(valid):
        valid = random.Random(args.seed).sample(valid, args.sample)

    print("total={} valid={} invalid={}".format(total, len(valid), len(errors)))
    for error in errors[:20]:
        print("ERROR {}".format(error), file=sys.stderr)
    if len(errors) > 20:
        print("ERROR ... {} more".format(len(errors) - 20), file=sys.stderr)
    if errors:
        return 1
    if not args.check_only:
        destination = Path(args.output).expanduser()
        if destination.suffix.lower() not in {".jsonl", ".ndjson"}:
            destination /= "prepared.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for item in valid:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("wrote={}".format(destination))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
