#!/usr/bin/env python3
"""Parse RSC payloads into per-metric CSVs.

Usage:
    uv run python experiments/parameter-estimation/rsc_parse.py \
        --rsc experiments/parameter-estimation/data/rsc_root.txt \
        --dump-keys

    uv run python experiments/parameter-estimation/rsc_parse.py \
        --rsc experiments/parameter-estimation/data/rsc_root.txt \
        --metric omniscience_accuracy=omniscience_breakdown.total.accuracy \
        --metric gdpval=gdpval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


# %%
def normalize_name(name: str) -> str:
    """Normalize model names/slugs for matching.

    Args:
        name: Raw model name or slug.

    Returns:
        Normalized slug.
    """
    name = name.lower().strip()
    name = name.replace("_", "-").replace(" ", "-")
    name = re.sub(r"[^a-z0-9-]+", "", name)
    name = re.sub(r"-+", "-", name)
    return name


def get_path(data: dict[str, Any], path: str) -> Any:
    """Traverse a dotted path inside a nested dict.

    Args:
        data: Nested dict payload.
        path: Dotted path to traverse.

    Returns:
        Value at the path or None if missing.
    """
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# %%
def _extract_json_arrays(text: str, key: str) -> list[list[Any]]:
    """Parse JSON arrays that follow a key inside a non-JSON text blob.

    Args:
        text: Raw RSC payload text.
        key: Key marker to anchor the array scan.

    Returns:
        Parsed JSON arrays for the given key marker.
    """
    arrays: list[list[Any]] = []
    start = text.find(key)
    while start != -1:
        idx = text.find("[", start)
        if idx == -1:
            break
        depth = 0
        in_string = False
        escape = False
        end = None
        for pos in range(idx, len(text)):
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            break
        array_text = text[idx : end + 1]
        try:
            parsed = json.loads(array_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            arrays.append(parsed)
        start = text.find(key, end + 1)
    return arrays


# %%
def _merge_key_tree(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge nested key trees in-place.

    Args:
        base: Base key tree to update.
        new: Key tree to merge into base.

    Returns:
        Updated base tree.
    """
    for key, value in new.items():
        if key not in base:
            base[key] = value
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge_key_tree(base[key], value)
            continue
        if isinstance(base[key], dict):
            # Prefer the existing dict to preserve nested keys.
            continue
        if isinstance(value, dict):
            # Promote to dict if any payload provides nested structure.
            base[key] = value
            continue
        if base[key] != value:
            base[key] = "mixed"
    return base


def _key_tree(value: Any) -> Any:
    """Build a key/type tree for a nested object.

    Args:
        value: Arbitrary nested value.

    Returns:
        A nested structure describing keys and value types.
    """
    if isinstance(value, dict):
        return {key: _key_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        if not value:
            return {"[]": "empty"}
        if all(isinstance(item, dict) for item in value):
            merged: dict[str, Any] = {}
            for item in value:
                _merge_key_tree(merged, _key_tree(item))
            return {"[]": merged}
        return {"[]": "primitive"}
    if value is None:
        return "null"
    return type(value).__name__


_MODEL_ARRAY_ANCHORS = ('"models":', '"data":', '"defaultData":')


def _iter_model_records(payload: str) -> Iterable[dict[str, Any]]:
    """Yield every model-shaped record across the supported anchor keys.

    AA's older RSC payloads kept per-model metrics inside ``models[]``; the
    current App Router pages put the same records under ``data[]`` (the
    full table) and ``defaultData[]`` (top-N filtered subsets), with
    snake_case field names rather than camelCase. We scan all anchors and
    yield every record — extract_metric_points handles deduplication by
    (model_name, metric value), so caller code can keep the first entry
    that carries a non-null value.
    """
    for anchor in _MODEL_ARRAY_ANCHORS:
        for arr in _extract_json_arrays(payload, anchor):
            for model in arr:
                if isinstance(model, dict):
                    yield model


def extract_model_key_tree(payloads: Iterable[str]) -> dict[str, Any]:
    """Build a merged key tree for model objects from RSC payloads.

    Args:
        payloads: Raw RSC payload texts.

    Returns:
        A merged key tree across all model objects.
    """
    merged: dict[str, Any] = {}
    for payload in payloads:
        for model in _iter_model_records(payload):
            _merge_key_tree(merged, _key_tree(model))
    return merged


# %%
def extract_metric_points(payloads: Iterable[str], metric_path: str) -> list[tuple[str, float, dict[str, Any]]]:
    """Extract (model_name, value, raw_model) for a metric path.

    Args:
        payloads: Raw RSC payload texts.
        metric_path: Dotted path to the numeric metric in the model object.

    Returns:
        Deduplicated list of (model_name, value, raw_model).
    """
    points: list[tuple[str, float, dict[str, Any]]] = []
    for payload in payloads:
        for model in _iter_model_records(payload):
            value = get_path(model, metric_path)
            if not isinstance(value, (int, float)):
                continue
            name = (
                model.get("shortName")
                or model.get("short_name")
                or model.get("name")
                or model.get("slug")
            )
            if not isinstance(name, str):
                continue
            points.append((name, float(value), model))
    deduped: dict[str, tuple[str, float, dict[str, Any]]] = {}
    for name, value, raw in points:
        key = normalize_name(name)
        if key not in deduped:
            deduped[key] = (name, value, raw)
    return list(deduped.values())


def parse_metrics(values: list[str]) -> list[tuple[str, str]]:
    """Parse metric definitions.

    Each metric can be either:
    - "path.to.value"
    - "label=path.to.value"

    Args:
        values: Metric definitions from CLI.

    Returns:
        A list of (label, path) tuples.
    """
    parsed: list[tuple[str, str]] = []
    for entry in values:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            path = entry
            label = entry.split(".")[-1]
        parsed.append((label, path))
    return parsed


_NEXT_F_MARKER = "self.__next_f.push(["


def _decode_next_f_chunks(text: str) -> str:
    """Pull RSC payload out of inline ``self.__next_f.push([N, "..."])`` calls.

    AA's App Router streams its server-component data inside ``__next_f.push``
    JS calls in the page HTML. Each call's second argument is a JSON-escaped
    string carrying one chunk of the payload. We concatenate the decoded
    chunks so the existing array-extraction logic can run on plain text.

    If the document does not contain any ``__next_f.push`` calls, returns the
    text unchanged so legacy ``_rsc=`` payloads keep working.

    Args:
        text: Raw payload text (HTML for App Router pages, raw RSC otherwise).

    Returns:
        Concatenated decoded chunks, or the original text if none were found.
    """
    if _NEXT_F_MARKER not in text:
        return text

    decoded: list[str] = []
    pos = 0
    while True:
        anchor = text.find(_NEXT_F_MARKER, pos)
        if anchor == -1:
            break
        # Skip past the array prefix and the leading numeric tag, e.g. ``1,``.
        cursor = anchor + len(_NEXT_F_MARKER)
        comma = text.find(",", cursor)
        if comma == -1:
            break
        # Find the JSON string that follows the comma.
        quote_start = text.find('"', comma)
        if quote_start == -1:
            break
        # Walk to the closing quote, respecting JSON-style backslash escapes.
        i = quote_start + 1
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                break
            i += 1
        if i >= len(text):
            break
        literal = text[quote_start : i + 1]
        try:
            decoded.append(json.loads(literal))
        except json.JSONDecodeError:
            pass
        pos = i + 1

    return "".join(decoded) if decoded else text


def load_payloads(paths: Iterable[Path]) -> list[str]:
    """Read RSC payload files and decode any inline ``__next_f.push`` chunks.

    Args:
        paths: Paths to RSC payload text files.

    Returns:
        Decoded payload text per input path.
    """
    payloads: list[str] = []
    for path in paths:
        payloads.append(_decode_next_f_chunks(path.read_text(encoding="utf-8")))
    return payloads


def write_csv(rows: list[tuple[str, float, dict[str, Any]]], label: str, output_path: Path) -> None:
    """Write metric rows to CSV.

    Args:
        rows: Extracted metric rows.
        label: Column label for the metric.
        output_path: Destination CSV path.
    """
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", label])
        for name, value, _raw in sorted(rows, key=lambda r: r[0].lower()):
            writer.writerow([name, f"{value:.4f}"])


# %%
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rsc",
        action="append",
        required=True,
        type=Path,
        help="Path to an RSC payload text file (repeatable).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help="Metric path (or label=path). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).parent / "data",
        type=Path,
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--dump-keys",
        action="store_true",
        help="Write model_keys.json for reference.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.rsc)
    if not args.metric and not args.dump_keys:
        raise ValueError("Provide at least one --metric or enable --dump-keys.")
    metrics = parse_metrics(args.metric or [])

    if metrics:
        for label, path in metrics:
            rows = extract_metric_points(payloads, path)
            if not rows:
                print(f"No rows extracted for metric: {path}")
                continue
            output_path = args.output_dir / f"{label}.csv"
            write_csv(rows, label, output_path)
            print(f"Wrote {len(rows)} rows to: {output_path}")

    if args.dump_keys:
        keys = extract_model_key_tree(payloads)
        if keys:
            keys_path = args.output_dir / "model_keys.json"
            keys_path.write_text(json.dumps(keys, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Wrote model keys to: {keys_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
