from site_search.chunking import partition_body


def _count_words(text: str) -> int:
    return len(text.split())


def test_whole_blocks_pack_under_target_including_context_cost() -> None:
    body = "one two\nthree four"

    chunks = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=6)

    assert chunks == [body]
    assert _count_words(f"Page Heading\n\n{chunks[0]}") == 6


def test_blocks_split_without_breaking_when_context_pushes_over_target() -> None:
    body = "one two\nthree four"

    chunks = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=5)

    assert chunks == ["one two\n", "three four"]


def test_oversized_block_splits_only_at_sentence_boundaries() -> None:
    body = "One two. Three four. Five six."

    chunks = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=5)

    assert chunks == ["One two. ", "Three four. ", "Five six."]
    assert "".join(chunks) == body


def test_oversized_single_sentence_remains_whole() -> None:
    body = "one two three four five six"

    chunks = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=4)

    assert chunks == [body]
    assert _count_words(f"Page Heading\n\n{chunks[0]}") > 4


def test_partition_is_exact_non_overlapping_and_ordered() -> None:
    body = "alpha beta\ngamma delta\nepsilon zeta"

    first = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=5)
    second = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=5)

    assert first == ["alpha beta\n", "gamma delta\n", "epsilon zeta"]
    assert "".join(first) == body
    assert second == first


def test_blank_separators_are_preserved_in_exact_partition() -> None:
    body = "a\n\n  \nb"

    chunks = partition_body(body, context="Page Heading", count_tokens=_count_words, token_target=3)

    assert "".join(chunks) == body
    assert chunks == ["a\n\n  \n", "b"]


def test_whitespace_only_body_emits_no_chunk() -> None:
    assert partition_body("\n  \n", context="Page Heading", count_tokens=_count_words) == []
