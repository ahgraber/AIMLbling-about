import logging
import os
from typing import FrozenSet, Iterable, List, Tuple

import psutil
from typing_extensions import override

from .base import LanguageIDModel

logger = logging.getLogger(__name__)


class LangID(LanguageIDModel):
    """Classify text with langid.

    Ref: https://github.com/saffsd/langid.py
    """

    name: str = "langid"

    def __init__(self):
        p = psutil.Process(os.getpid())
        mem_before = p.memory_info().vms  # .rss

        self.load()
        _ = self.predict("hello darkness my ol' fren")

        self.memory = (p.memory_info().vms - mem_before) / self._MB

        super().__init__()  # may require model to be loaded

    @override
    def load(self):
        from langid.langid import LanguageIdentifier, model

        self.model = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    @override
    def _get_labels(self) -> FrozenSet[str]:
        # ref: https://github.com/saffsd/langid.py/tree/master
        # typos:off
        supported_languages = [
            "af",
            "am",
            "an",
            "ar",
            "as",
            "az",
            "be",
            "bg",
            "bn",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "dz",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "ga",
            "gl",
            "gu",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jv",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "ku",
            "ky",
            "la",
            "lb",
            "lo",
            "lt",
            "lv",
            "mg",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "nb",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "or",
            "pa",
            "pl",
            "ps",
            "pt",
            "qu",
            "ro",
            "ru",
            "rw",
            "se",
            "si",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "ug",
            "uk",
            "ur",
            "vi",
            "vo",
            "wa",
            "xh",
            "zh",
            "zu",
        ]
        # typos:on
        return frozenset(sorted(self.label_to_lang(label) for label in supported_languages))

    @override
    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        if isinstance(text, str):
            label, prob = self.model.classify(text)
            lang = self.label_to_lang(label)
            return lang if not with_prob else (lang, prob)
        else:
            raise TypeError(f"Provided {type(text)=}; must be string.")

    @override
    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        if isinstance(texts, Iterable) and not isinstance(texts, str):
            return [self.predict(text, with_prob) for text in texts]
        else:
            raise TypeError(f"Provided {type(texts)=}; must be iterable (list).")
