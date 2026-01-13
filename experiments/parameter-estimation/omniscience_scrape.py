#!/usr/bin/env python3
"""Scrape omniscience accuracy values (with tooltip precision) from artificialanalysis.ai.

This notebook-style script opens the page in a headless browser and extracts the underlying
plot data (Highcharts/Plotly) or network responses, then writes a CSV/JSON of
model -> accuracy values with full precision.
"""

# %%
from __future__ import annotations

import asyncio
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import nest_asyncio
from playwright.async_api import async_playwright

from aiml.utils import ensure_playwright_chromium

# %%
# user parameters
# How this scrape works (high level):
# 1) Launch Playwright and load the omniscience page.
# 2) Try to read chart data directly from Highcharts/Plotly in the browser.
# 3) Fallback to JSON embedded in the DOM or captured network responses.
# 4) Final fallback: parse Next.js RSC payloads that include model data.
URL = "https://artificialanalysis.ai/?omniscience=omniscience-accuracy&model-filters=frontier-model&models=claude-opus-4-5-thinking%2Cgemini-3-pro%2Cgpt-5-1%2Cgemini-3-flash-reasoning%2Cgpt-5-2-medium%2Cclaude-4-5-sonnet-thinking%2Cglm-4-7%2Cgpt-5-medium%2Cgrok-4%2Cdeepseek-v3-2-reasoning%2Co3%2Cgemini-3-pro-low%2Ckimi-k2-thinking%2Cminimax-m2-1%2Cgpt-5-mini-medium%2Cclaude-4-5-haiku-reasoning%2Cminimax-m2%2Cnova-2-0-pro-reasoning-medium%2Cclaude-3-7-sonnet-thinking%2Cgemini-2-5-pro%2Cgpt-oss-120b%2Cglm-4-6-reasoning%2Cnova-2-0-lite-reasoning-medium%2Cqwen3-235b-a22b-instruct-2507-reasoning%2Cnova-2-0-omni-reasoning-medium%2Cgemini-2-5-flash-reasoning%2Cdeepseek-r1%2Cglm-4.5%2Cgpt-5-nano-medium%2Cgpt-4-1%2Cgrok-3%2Cgpt-oss-20b%2Cnvidia-nemotron-3-nano-30b-a3b-reasoning%2Cgpt-oss-120b-low%2Cqwen3-30b-a3b-2507-reasoning%2Cdeepseek-v3-0324%2Cgpt-oss-20b-low%2Cllama-4-maverick%2Cqwen3-14b-instruct-reasoning%2Cministral-14b%2Cqwen3-30b-a3b-instruct-reasoning%2Cgpt-4o%2Cllama-4-scout%2Ccommand-a%2Cgemma-3-27b%2Cgemma-3-12b%2Cgemma-3-1b%2Cgemma-3-4b%2Colmo-3-1-32b-think#aa-omniscience-accuracy"
CSV_PATH = None
JSON_PATH = None
TIMEOUT_MS = 60000
HEADLESS = os.getenv("AIML_HEADLESS", "true").lower() not in {"0", "false", "no"}
WAIT_RESPONSE_SUBSTRINGS = ("omniscience", "artificialanalysis")
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RSC_URL_SUBSTRING = "/evaluations/omniscience?_rsc="
TARGET_VALUE_PATH = os.getenv("AIML_VALUE_PATH", "omniscience_breakdown.total.accuracy")
TARGET_VALUE_LABEL = os.getenv("AIML_VALUE_LABEL", "omniscience_accuracy")
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / f"{TARGET_VALUE_LABEL}.csv"
# Use AIML_VALUE_PATH to target a different metric in the model record,
# for example: "gdpval" or "omniscience_breakdown.total.hallucination_rate".
DUMP_KEY_TREE = os.getenv("AIML_DUMP_KEYS", "false").lower() in {"1", "true", "yes"}


# %%
@dataclass(frozen=True)
class DataPoint:
    model: str
    value: float
    raw: dict[str, Any]


MODEL_KEYS = ("model", "modelName", "name", "model_name", "label")
VALUE_KEYS = (
    "accuracy",
    "value",
    "score",
    "omniscience_accuracy",
    "omniscienceAccuracy",
    "metricValue",
    "metric_value",
)


# %%
def normalize_name(name: str) -> str:
    """Normalize model names/slugs for matching.

    Args:
        name: Raw model name or slug from a payload or URL.

    Returns
    -------
        A lowercase, dash-separated slug suitable for comparisons.
    """
    name = name.lower().strip()
    name = name.replace("_", "-").replace(" ", "-")
    name = re.sub(r"[^a-z0-9-]+", "", name)
    name = re.sub(r"-+", "-", name)
    return name


def parse_models_from_url(url: str) -> set[str]:
    """Extract normalized model slugs from the query string.

    Args:
        url: Target URL that may include a comma-separated models parameter.

    Returns
    -------
        A set of normalized model slugs for filtering scraped data.
    """
    query = parse_qs(urlparse(url).query)
    models = query.get("models", [])
    if not models:
        return set()
    split_models: list[str] = []
    for entry in models:
        split_models.extend(entry.split(","))
    return {normalize_name(m) for m in split_models if m}


def get_path(data: dict[str, Any], path: str) -> Any:
    """Traverse a dotted path inside a nested dict."""
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def maybe_point(obj: dict[str, Any]) -> tuple[str, float] | None:
    """Try to interpret a dict as a single model/value datapoint.

    Args:
        obj: Arbitrary JSON object from payloads or chart data.

    Returns
    -------
        A (model_name, value) tuple if keys match known patterns; otherwise None.
    """
    model_val = None
    for key in MODEL_KEYS:
        if key in obj and isinstance(obj[key], str):
            model_val = obj[key]
            break
    if not model_val:
        return None
    for key in VALUE_KEYS:
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return model_val, float(value)
    value = get_path(obj, TARGET_VALUE_PATH)
    if isinstance(value, (int, float)):
        return model_val, float(value)
    return None


def find_points(payload: Any) -> list[DataPoint]:
    """Walk a JSON payload and collect all candidate datapoints.

    Args:
        payload: Any JSON-compatible structure (dict/list/primitive).

    Returns
    -------
        A list of DataPoint items derived from matching shapes.
    """
    points: list[DataPoint] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            candidate = maybe_point(node)
            if candidate:
                model, value = candidate
                points.append(DataPoint(model=model, value=value, raw=node))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return points


def _extract_json_arrays(text: str, key: str) -> list[list[Any]]:
    """Parse JSON arrays that follow a key inside a non-JSON text blob.

    This is used for Next.js RSC responses, which interleave JSON with
    React serialization. We scan for a key, then parse the next bracketed
    array while honoring string escapes to avoid premature termination.

    Args:
        text: Raw response text (not necessarily valid JSON).
        key: Key marker to anchor the array scan (e.g., '"models":').

    Returns
    -------
        A list of JSON arrays parsed successfully from the blob.
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


def extract_points_from_rsc(payloads: Iterable[str], model_slugs: set[str]) -> list[DataPoint]:
    """Extract omniscience accuracy from Next.js RSC payloads.

    Args:
        payloads: Raw text payloads returned by the _rsc route.
        model_slugs: Optional normalized slugs to filter the output.

    Returns
    -------
        A list of DataPoint entries for omniscience accuracy.
    """
    points: list[DataPoint] = []
    for payload in payloads:
        for models in _extract_json_arrays(payload, '"models":'):
            for model in models:
                if not isinstance(model, dict):
                    continue
                value = get_path(model, TARGET_VALUE_PATH)
                if not isinstance(value, (int, float)):
                    continue
                name = model.get("short_name") or model.get("name") or model.get("slug")
                if not isinstance(name, str):
                    continue
                slug = model.get("slug")
                if isinstance(slug, str):
                    candidate = slug
                else:
                    candidate = name
                if model_slugs and normalize_name(candidate) not in model_slugs:
                    continue
                points.append(DataPoint(model=name, value=float(value), raw=model))
    return dedupe(points)


def _merge_key_tree(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for key, value in new.items():
        if key not in base:
            base[key] = value
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge_key_tree(base[key], value)
        elif base[key] != value:
            base[key] = "mixed"
    return base


def _key_tree(value: Any) -> Any:
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


def extract_model_key_tree(payloads: Iterable[str]) -> dict[str, Any]:
    """Build a merged key tree for model objects from RSC payloads."""
    merged: dict[str, Any] = {}
    for payload in payloads:
        for models in _extract_json_arrays(payload, '"models":'):
            for model in models:
                if not isinstance(model, dict):
                    continue
                _merge_key_tree(merged, _key_tree(model))
    return merged


def dedupe(points: Iterable[DataPoint]) -> list[DataPoint]:
    """Deduplicate points by normalized model name, keeping first seen."""
    seen: dict[str, DataPoint] = {}
    for point in points:
        key = normalize_name(point.model)
        if key not in seen:
            seen[key] = point
    return list(seen.values())


async def extract_plot_data(page) -> list[dict[str, Any]]:
    """Extract data from Highcharts/Plotly on the page.

    This executes in the browser context (page.evaluate) and reads any
    live chart objects to capture the underlying plotted values.
    """
    return await page.evaluate(
        """
        () => {
          const results = [];
          if (window.Highcharts && Array.isArray(window.Highcharts.charts)) {
            for (const chart of window.Highcharts.charts) {
              if (!chart || !chart.series) continue;
              for (const series of chart.series) {
                if (!series || !series.points) continue;
                for (const point of series.points) {
                  results.push({
                    source: "highcharts",
                    chartId: chart.renderTo && chart.renderTo.id,
                    series: series.name,
                    name: point.name || point.category || point.id || null,
                    x: point.x,
                    y: point.y,
                    raw: {
                      id: point.id,
                      name: point.name,
                      category: point.category,
                      custom: point.custom,
                    },
                  });
                }
              }
            }
          }
          const plotElements = document.querySelectorAll('.js-plotly-plot');
          for (const plotEl of plotElements) {
            const data = plotEl.data || plotEl._fullData || [];
            for (const trace of data) {
              const y = trace.y || trace.values || [];
              const x = trace.x || [];
              const text = trace.text || trace.hovertext || [];
              const labels = trace.labels || [];
              for (let i = 0; i < Math.max(y.length, x.length, text.length, labels.length); i++) {
                results.push({
                  source: "plotly",
                  chartId: plotEl.id || null,
                  series: trace.name,
                  name: text[i] || labels[i] || trace.name || null,
                  x: x[i],
                  y: y[i],
                  raw: {
                    text: text[i],
                    label: labels[i],
                    customdata: trace.customdata ? trace.customdata[i] : null,
                  },
                });
              }
            }
          }
          return results;
        }
        """
    )


async def extract_embedded_payloads(page) -> list[Any]:
    """Pull embedded JSON blobs that frameworks often inject into the DOM."""
    return await page.evaluate(
        """
        () => {
          const payloads = [];
          const maybePush = (value) => {
            if (value && typeof value === "object") {
              payloads.push(value);
            }
          };
          maybePush(window.__NEXT_DATA__);
          maybePush(window.__NUXT__);
          maybePush(window.__APOLLO_STATE__);
          maybePush(window.__INITIAL_STATE__);
          const nextEl = document.getElementById("__NEXT_DATA__");
          if (nextEl && nextEl.textContent) {
            try {
              payloads.push(JSON.parse(nextEl.textContent));
            } catch (err) {
              // ignore parse failures
            }
          }
          const jsonScripts = document.querySelectorAll('script[type="application/json"]');
          for (const script of jsonScripts) {
            if (!script.textContent) continue;
            try {
              payloads.push(JSON.parse(script.textContent));
            } catch (err) {
              // ignore parse failures
            }
          }
          return payloads;
        }
        """
    )


def filter_points(points: list[dict[str, Any]], model_slugs: set[str]) -> list[DataPoint]:
    """Filter chart points to the models in the URL (if present)."""
    results: list[DataPoint] = []
    for point in points:
        value = point.get("y")
        if not isinstance(value, (int, float)):
            continue
        name = point.get("name") or point.get("series")
        if not isinstance(name, str):
            continue
        if model_slugs:
            if normalize_name(name) not in model_slugs:
                continue
        results.append(DataPoint(model=name, value=float(value), raw=point))
    return dedupe(results)


def write_outputs(points: list[DataPoint], csv_path: str | None, json_path: str | None) -> None:
    """Write CSV/JSON outputs if paths are provided."""
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", TARGET_VALUE_LABEL])
            for point in sorted(points, key=lambda p: p.model.lower()):
                writer.writerow([point.model, f"{point.value:.4f}"])
    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "model": point.model,
                        "accuracy": point.value,
                        "raw": point.raw,
                    }
                    for point in points
                ],
                handle,
                indent=2,
                sort_keys=True,
            )


def run_async(coro: asyncio.Future):
    """Run a coroutine in notebooks or scripts without event loop errors."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    nest_asyncio.apply()
    return loop.run_until_complete(coro)


async def capture_response(response, payloads: list[Any]) -> None:
    """Capture likely data-bearing responses for fallback parsing."""
    content_type = response.headers.get("content-type", "")
    url = response.url.lower()
    should_capture = "application/json" in content_type or any(s in url for s in WAIT_RESPONSE_SUBSTRINGS)
    if not should_capture:
        return
    if RSC_URL_SUBSTRING in url:
        # Next.js RSC streams JSON-like data inside a custom protocol.
        # We save the raw text for offline parsing.
        try:
            text = await response.text()
            captured_rsc_payloads.append(text)
        except Exception:
            pass
        return
    try:
        payloads.append(await response.json())
    except Exception:
        try:
            text = await response.text()
            payloads.append(json.loads(text))
        except Exception:
            return


# %%
if not URL:
    raise ValueError("Set URL to an artificialanalysis.ai omniscience page before running.")

chromium_path = ensure_playwright_chromium()
print(f"Using Playwright Chromium at: {chromium_path}")

model_slugs = parse_models_from_url(URL)

captured_payloads: list[Any] = []
captured_rsc_payloads: list[str] = []


async def scrape_page() -> list[dict[str, Any]]:
    """Open the page, wait for charts/data, and capture plot points."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page()
        tasks: list[asyncio.Task] = []

        def handle_response(response) -> None:
            # Capture in the background to avoid blocking page rendering.
            tasks.append(asyncio.create_task(capture_response(response, captured_payloads)))

        page.on("response", handle_response)
        await page.goto(URL, wait_until="networkidle", timeout=TIMEOUT_MS)
        # Some data loads lazily after navigation; wait for any relevant response.
        try:
            await page.wait_for_response(
                lambda response: response.status == 200
                and any(substr in response.url.lower() for substr in WAIT_RESPONSE_SUBSTRINGS),
                timeout=TIMEOUT_MS,
            )
        except Exception:
            pass
        try:
            # If charts are present, give the page a chance to render them.
            await page.wait_for_function(
                "() => (window.Highcharts && window.Highcharts.charts && window.Highcharts.charts.length) || document.querySelector('.js-plotly-plot')",
                timeout=TIMEOUT_MS,
            )
        except Exception:
            pass
        await page.wait_for_timeout(3000)

        plot_points = await extract_plot_data(page)
        embedded_payloads = await extract_embedded_payloads(page)
        await browser.close()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if embedded_payloads:
            captured_payloads.extend(embedded_payloads)

    return plot_points


plot_points = run_async(scrape_page())

points = filter_points(plot_points, model_slugs)
if not points:
    # Fallback: scan captured JSON payloads for model/value pairs.
    payload_points: list[DataPoint] = []
    for payload in captured_payloads:
        payload_points.extend(find_points(payload))
    if model_slugs:
        payload_points = [point for point in payload_points if normalize_name(point.model) in model_slugs]
    points = dedupe(payload_points)
else:
    payload_points = []

rsc_points = extract_points_from_rsc(captured_rsc_payloads, model_slugs)
if rsc_points:
    # Merge RSC-derived points with any earlier matches.
    points = dedupe([*points, *rsc_points])

if DUMP_KEY_TREE:
    model_key_tree = extract_model_key_tree(captured_rsc_payloads)
    if model_key_tree:
        keys_path = OUTPUT_DIR / "model_keys.json"
        keys_path.write_text(json.dumps(model_key_tree, indent=2, sort_keys=True), encoding="utf-8")

if not points:
    print(
        "No points were extracted. Try running with a visible browser or inspect network responses.",
        file=sys.stderr,
    )
    raise RuntimeError("No points extracted")

# %%
write_outputs(points, DEFAULT_OUTPUT_CSV, JSON_PATH)

for point in sorted(points, key=lambda p: p.model.lower()):
    print(f"{point.model}: {point.value:.4f}")

grouped: dict[str, list[DataPoint]] = defaultdict(list)
for point in points:
    grouped[normalize_name(point.model)].append(point)

duplicates = {k: v for k, v in grouped.items() if len(v) > 1}
if duplicates:
    print("\nDuplicate normalized names detected (review JSON output):", file=sys.stderr)
    for name in duplicates:
        print(f"- {name}", file=sys.stderr)

# %%
