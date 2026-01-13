#!/usr/bin/env python3
"""Fetch Next.js RSC payloads for a URL and write them to a text file.

Usage:
    uv run python experiments/parameter-estimation/rsc_fetch.py \
        --url <URL> \
        --output-dir "experiments/parameter-estimation/data/"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import nest_asyncio
from playwright.async_api import async_playwright

from aiml.utils import ensure_playwright_chromium

# %%
RSC_QUERY_MARKER = "_rsc="
DEFAULT_TIMEOUT_MS = 60000


def _default_output_path(url: str, output_dir: Path) -> Path:
    """Build a stable output filename from the URL path.

    Args:
        url: Page URL used to derive a stable output name.
        output_dir: Base directory for outputs.

    Returns:
        Path to the default output file.
    """
    path = urlparse(url).path.strip("/") or "root"
    slug = path.replace("/", "_")
    return output_dir / f"rsc_{slug}.txt"


# %%
async def fetch_rsc_payloads(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> list[str]:
    """Open a page and capture Next.js RSC responses.

    Args:
        url: Page URL to load.
        timeout_ms: Timeout for navigation and response waits.

    Returns:
        A list of raw RSC response bodies.
    """
    payloads: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        tasks: list[asyncio.Task] = []

        def handle_response(response) -> None:
            if RSC_QUERY_MARKER in response.url:
                tasks.append(asyncio.create_task(response.text()))

        page.on("response", handle_response)
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        try:
            await page.wait_for_response(lambda response: RSC_QUERY_MARKER in response.url, timeout=timeout_ms)
        except Exception:
            pass
        await page.wait_for_timeout(2000)
        await browser.close()

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, str):
                    payloads.append(result)

    return payloads


def _join_payloads(payloads: Iterable[str]) -> str:
    """Combine multiple payloads with a readable delimiter.

    Args:
        payloads: RSC response bodies to join.

    Returns:
        Combined payload text with delimiters.
    """
    parts = []
    for idx, payload in enumerate(payloads, start=1):
        parts.append(f"\n--- rsc payload {idx} ---\n{payload}\n")
    return "\n".join(parts).strip()


# %%
def run_async(coro: asyncio.Future):
    """Run a coroutine in notebooks or scripts without event loop errors.

    Args:
        coro: Coroutine to execute.

    Returns:
        Result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    nest_asyncio.apply()
    return loop.run_until_complete(coro)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Page URL to load.")
    parser.add_argument(
        "--output",
        help="Output file path for the RSC payloads.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).parent / "data",
        type=Path,
        help="Directory for output files when --output is not set.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Timeout for navigation and response waits.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    args = parse_args()
    ensure_playwright_chromium()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else _default_output_path(args.url, args.output_dir)
    payloads = run_async(fetch_rsc_payloads(args.url, args.timeout_ms))
    if not payloads:
        raise RuntimeError("No RSC payloads captured.")
    output_path.write_text(_join_payloads(payloads), encoding="utf-8")
    print(f"Wrote {len(payloads)} payload(s) to: {output_path}")
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
