from __future__ import annotations

from pathlib import Path

from model2vec import StaticModel

from site_search.export import resolve_tokenizer_config, tokenize_from_config


def test_resolved_tokenizer_config_reproduces_reference(potion_model_path: Path) -> None:
    provider = StaticModel.from_pretrained(potion_model_path, dimensionality=128)
    config = resolve_tokenizer_config(provider.tokenizer.to_str(), provider.unk_token_id)
    samples = ["semantic search", "café naïve", "semantic xyzzyplugh☃"]

    assert config["tokenizer"]["normalizer"]["strip_accents"] is True
    assert config["add_special_tokens"] is False
    assert config["drop_unknown"] is True
    assert [tokenize_from_config(text, config) for text in samples] == provider.tokenize(samples)
    special_ids = {
        provider.tokenizer.token_to_id("[CLS]"),
        provider.tokenizer.token_to_id("[SEP]"),
    }
    assert all(special_ids.isdisjoint(tokenize_from_config(text, config)) for text in samples)
