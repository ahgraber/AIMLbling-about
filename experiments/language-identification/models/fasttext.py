import logging
import os
from typing import FrozenSet, Iterable, List, Tuple

from huggingface_hub import hf_hub_download
from langcodes import Language, standardize_tag
import psutil
from typing_extensions import override

from .base import LanguageIDModel

logger = logging.getLogger(__name__)


class FastText(LanguageIDModel):
    """Classify text with FastText.

    FastText is deprecated as of 16 March 2024.
    Ref: https://github.com/facebookresearch/fastText/tree/main
    """

    name: str = "FastText"

    def __init__(self):
        p = psutil.Process(os.getpid())
        mem_before = p.memory_info().vms  # .rss

        self.load()
        _ = self.predict("hello darkness my ol' fren")

        self.memory = (p.memory_info().vms - mem_before) / self._MB
        # self.memory = (p.memory_info().rss - mem_before) / self._MB

        super().__init__()  # may require model to be loaded

    @override
    def load(self):
        import fasttext

        local_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(local_path)

    @override
    def label_to_lang(self, label: str) -> str:
        """Return standardized tag form of language."""
        tag = Language.get(standardize_tag(label.removeprefix("__label__"), macro=True)).language
        return tag

    @override
    def _get_labels(self) -> FrozenSet[str]:
        supported_languages = self.model.get_labels()
        return frozenset(sorted(self.label_to_lang(label) for label in supported_languages))

    @override
    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        if isinstance(text, str):
            text = text.replace("\n", "  ")  # fasttext doesn't like newlines
            label, prob = self.model.predict(text, k=1)
            lang = self.label_to_lang(label[0])
            return lang if not with_prob else (lang, prob[0])
        else:
            raise TypeError(f"Provided {type(text)=}; must be string.")

    @override
    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        if isinstance(texts, Iterable) and not isinstance(texts, str):
            labels, probs = self.model.predict(texts, k=1)
            langs = [self.label_to_lang(label[0]) for label in labels]
            probs = [float(p[0]) for p in probs]
            return langs if not with_prob else list(zip(langs, probs))
        else:
            raise TypeError(f"Provided {type(texts)=}; must be iterable (list).")
