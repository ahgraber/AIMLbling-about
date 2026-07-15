from __future__ import annotations

from collections.abc import Callable, Sequence
import re

TOKEN_TARGET = 256
_SENTENCE = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", re.DOTALL)


def _embedding_text(context: str, body: str) -> str:
    return f"{context}\n\n{body}"


def _pack_units(
    units: Sequence[str], *, context: str, count_tokens: Callable[[str], int], token_target: int
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if current and count_tokens(_embedding_text(context, candidate)) > token_target:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _sentence_units(block: str) -> list[str]:
    units = [match.group(0) for match in _SENTENCE.finditer(block)]
    return units if "".join(units) == block else [block]


def partition_body(
    body: str,
    *,
    context: str,
    count_tokens: Callable[[str], int],
    token_target: int = TOKEN_TARGET,
) -> list[str]:
    """Partition flattened section text without overlap or arbitrary cuts."""
    if token_target <= 0:
        raise ValueError("token_target must be positive")
    blocks: list[str] = []
    leading_separator = ""
    for line in body.splitlines(keepends=True):
        if line.strip():
            blocks.append(leading_separator + line)
            leading_separator = ""
        elif blocks:
            blocks[-1] += line
        else:
            leading_separator += line

    chunks: list[str] = []
    pending: list[str] = []
    for block in blocks:
        candidate = "".join([*pending, block])
        if count_tokens(_embedding_text(context, candidate)) <= token_target:
            pending.append(block)
            continue
        if pending:
            chunks.append("".join(pending))
            pending = []
        if count_tokens(_embedding_text(context, block)) <= token_target:
            pending = [block]
        else:
            chunks.extend(
                _pack_units(
                    _sentence_units(block), context=context, count_tokens=count_tokens, token_target=token_target
                )
            )
    if pending:
        chunks.append("".join(pending))
    return chunks
