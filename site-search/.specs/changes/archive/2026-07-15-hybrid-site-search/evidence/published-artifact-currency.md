# Podman builds ship current search artifacts

Recorded: 2026-07-13

## The clean image contains the complete artifact set

The test used the user-owned Podman baseline at commit `00ca612667cd6a1f869207bea7cfe716d4a87709` and a running AppleHV machine.
The native arm64 build exercised the repository's supported entrypoint:

```text
nix develop -c just hugo build linux/arm64
```

The first attempt exposed a clean-context failure because `.dockerignore` excluded the Dockerfile's `experiments/aiml/pyproject.toml` copy source.
A temporary standalone-context check proved that `pyproject.toml`, `uv.lock`, and the `site-search` project files are sufficient for `uv sync --frozen --package site-search --no-default-groups`.
The Dockerfile removed the unnecessary copy, and a regression test prevents that excluded path from returning.

The corrected build completed both Hugo passes, passed the corpus gate, generated the semantic index between the passes, and tagged `ghcr.io/ahgraber/aimlbling-about:debug`.
Inspection of the final nginx stage returned these filesystem byte counts:

| Artifact                             |          Bytes |
| ------------------------------------ | -------------: |
| `/site/search/token-table.bin`       |      7,559,168 |
| `/site/search/token-scales.bin`      |        118,112 |
| `/site/search/doc-vectors.bin`       |      1,141,248 |
| `/site/search/manifest.json`         |      1,455,612 |
| `/site/search/tokenizer-config.json` |        446,398 |
| `/site/search/meta.json`             |            327 |
| **Total**                            | **10,720,865** |

The shipped `meta.json` reported 3,298 corpus sections, 4,458 chunks, 256 dimensions, zero exclusions, and the same 10,720,865-byte total.
The baseline manifest did not contain the regeneration canary route.

## A content change regenerates the image manifest

The second build added a temporary published page at `/blog/search-artifact-currency-fixture/` with the unique text `quokka-lantern regeneration canary`.
The same native recipe rebuilt the image from that working tree.

The changed build reported:

- 409 Hugo pages instead of 408
- 3,300 corpus sections instead of 3,298
- 4,460 semantic chunks instead of 4,458
- 10,722,011 artifact bytes instead of 10,720,865

Final-stage inspection found `search-artifact-currency-fixture` entries in `/site/search/manifest.json`, including the `regeneration-canary` section chunk.
This proves the published artifact set came from the changed build's Hugo corpus rather than a stale host artifact.

The temporary page was deleted after inspection.
The local debug tag was restored to the clean baseline image, whose manifest lacks the canary route and whose metadata again reports 3,298 sections and 4,458 chunks.

## Validation commands passed

```text
uv run --package site-search pytest site-search/tests/test_build_pipeline.py -q
podman run --rm --entrypoint /bin/sh ghcr.io/ahgraber/aimlbling-about:debug ...
```

The focused Python suite passed all three build-pipeline tests.
Both image inspections exited successfully.
