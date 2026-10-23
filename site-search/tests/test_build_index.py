from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece

import numpy as np
import numpy.typing as npt

from site_search.build_index import BuildSummary, CorpusRejectedError, build_index, main
from site_search.embedding import embed_token_ids
from site_search.eval import keyword_documents_from_corpus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class FakeEmbeddingModel:
    model_id = "fake-static-model"
    unknown_token_id = 0

    def __init__(self) -> None:
        vocab = {"[UNK]": 0, "plain": 1, "body": 2, "known": 3, "intro": 4}
        tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token=next(iter(vocab))))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer = tokenizer
        self.token_vectors = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32
        )
        self.embedded_texts: list[str] = []

    @property
    def tokenizer_json(self) -> str:
        return self._tokenizer.to_str()

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        return [
            [token_id for token_id in self._tokenizer.encode(text, add_special_tokens=False).ids if token_id != 0]
            for text in texts
        ]

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        self.embedded_texts.extend(texts)
        return embed_token_ids(self.tokenize(texts), self.token_vectors)


def _write_corpus(path: Path, data: dict[str, str]) -> None:
    path.write_text(json.dumps({"/blog/example/": {"title": "Example", "data": data}}), encoding="utf-8")


def test_rejected_corpus_does_not_modify_artifact_output(tmp_path: Path) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"plain#Plain": "plain body", "named#Suspicious": ""})
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "meta.json").write_text('{"complete":true}', encoding="utf-8")
    model_loaded = False

    def model_factory() -> FakeEmbeddingModel:
        nonlocal model_loaded
        model_loaded = True
        return FakeEmbeddingModel()

    with pytest.raises(CorpusRejectedError, match=r"gate: FAIL"):
        build_index(search_data, output, model_factory=model_factory)

    assert model_loaded is False
    assert [path.name for path in output.iterdir()] == ["meta.json"]
    assert (output / "meta.json").read_text() == '{"complete":true}'


def test_builder_records_section_and_empty_counts_in_metadata(tmp_path: Path) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"plain#Plain": "plain body", "known#Known": "known body", "parent#Parent": ""})
    output = tmp_path / "artifacts"

    build_index(search_data, output, model_factory=FakeEmbeddingModel)

    meta = json.loads((output / "meta.json").read_text())
    assert meta["corpus_stats"] == {"section_count": 3, "empty_count": 1}


def test_builder_embeds_heading_isolated_context_in_order(tmp_path: Path) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"first#Plain": "plain body", "second#Known": "known body"})
    output = tmp_path / "artifacts"
    model = FakeEmbeddingModel()

    summary = build_index(search_data, output, model_factory=lambda: model)

    assert isinstance(summary, BuildSummary)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [(row["page_id"], row["url"], row["title"], row["heading"]) for row in manifest] == [
        ("/blog/example", "/blog/example#first", "Example", "Plain"),
        ("/blog/example", "/blog/example#second", "Example", "Known"),
    ]
    assert model.embedded_texts == ["Example\nPlain\n\nplain body", "Example\nKnown\n\nknown body"]
    assert "known body" not in model.embedded_texts[0]
    assert "plain body" not in model.embedded_texts[1]
    assert summary.gate_result.passed


def test_intro_context_does_not_repeat_page_title(tmp_path: Path) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"": "plain body"})
    model = FakeEmbeddingModel()

    build_index(search_data, tmp_path / "artifacts", model_factory=lambda: model)

    assert model.embedded_texts == ["Example\n\nplain body"]


def test_eval_keyword_documents_join_the_manifest_on_page_id_and_url(tmp_path: Path) -> None:
    # The eval fuses its keyword page ids with manifest page ids by equality, so
    # both must come from the same derivation.
    search_data = tmp_path / "search-data.json"
    search_data.write_text(
        json.dumps(
            {
                "/": {"title": "Home", "data": {"": "known body"}},
                "/blog/example/": {"title": "Example", "data": {"": "plain body", "s#Known": "known body"}},
                "/glossary": {"title": "Glossary", "data": {"t#Term": "plain body"}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    build_index(search_data, output, model_factory=FakeEmbeddingModel)

    manifest = json.loads((output / "manifest.json").read_text())
    documents = keyword_documents_from_corpus(search_data)
    assert sorted((document.page_id, document.url) for document in documents) == sorted(
        (row["page_id"], row["url"]) for row in manifest
    )


def test_builder_derives_intro_page_identity_from_corpus(tmp_path: Path) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"": "known body"})
    output = tmp_path / "artifacts"

    build_index(search_data, output, model_factory=FakeEmbeddingModel)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest == [
        {
            "chunk_id": manifest[0]["chunk_id"],
            "page_id": "/blog/example",
            "url": "/blog/example",
            "title": "Example",
            "heading": "Example",
        },
    ]


def test_cli_uses_arguments_and_reports_exact_artifact_sizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    search_data = tmp_path / "search-data.json"
    _write_corpus(search_data, {"plain#Plain": "plain body"})
    output = tmp_path / "artifacts"

    exit_code = main(
        ["--search-data", str(search_data), "--out", str(output)],
        model_factory=FakeEmbeddingModel,
    )

    assert exit_code == 0
    assert (output / "meta.json").exists()
    captured = capsys.readouterr()
    size_lines = [line for line in captured.out.splitlines() if line.startswith("artifact-size ")]
    reported = {
        fields[0].removeprefix("name="): int(fields[1].removeprefix("bytes="))
        for line in size_lines
        if len(fields := line.split()[1:]) == 2
    }
    expected = {path.name: path.stat().st_size for path in output.iterdir() if path.is_file()}
    assert reported == expected
    assert f"artifact-total bytes={sum(expected.values())}" in captured.out
