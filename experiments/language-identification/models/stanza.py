import logging
import os
from typing import FrozenSet, Iterable, List, Tuple

import psutil
from typing_extensions import override

from .base import LanguageIDModel

logger = logging.getLogger(__name__)


class StanzaNLP(LanguageIDModel):
    """Classify text with FastText.

    Ref: https://stanfordnlp.github.io/stanza/langid.html
    """

    name = "stanza"

    def __init__(self):
        p = psutil.Process(os.getpid())
        mem_before = p.memory_info().vms  # .rss

        self.load()
        _ = self.predict("hello darkness my ol' fren")

        self.memory = (p.memory_info().vms - mem_before) / self._MB

        super().__init__()  # may require model to be loaded

    @override
    def load(self):
        import stanza
        from stanza.pipeline.core import Pipeline

        from aiml.utils import torch_device

        stanza.download(lang="multilingual")

        self.model = Pipeline(lang="multilingual", processors="langid", device=torch_device())

    @override
    def _get_labels(self) -> FrozenSet[str]:
        # Ref: https://stanfordnlp.github.io/stanza/langid.html
        # typos:off
        supported_languages = "af ar be bg bxr ca cop cs cu da de el en es et eu fa fi fr fro ga gd gl got grc he hi hr hsb hu hy id it ja kk kmr ko la lt lv lzh mr mt nl nn no olo orv pl pt ro ru sk sl sme sr sv swl ta te tr ug uk ur vi wo zh-hans zh-hant".split()
        # typos:on
        return frozenset(sorted(self.label_to_lang(label) for label in supported_languages))

    @override
    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        from stanza.models.common.doc import Document

        # probabilities aren't accessible
        prob = 1.0
        if with_prob:
            logger.info("Stanza does not support probabilities; returning all as 1")
        if isinstance(text, str):
            doc = Document([], text=text)
            self.model(doc)
            lang = self.label_to_lang(doc.lang)
            return lang if not with_prob else (lang, prob)
        else:
            raise TypeError(f"Provided {type(text)=}; must be string.")

    @override
    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        from stanza.models.common.doc import Document

        # probabilities aren't accessible
        prob = 1.0
        if with_prob:
            logger.info("Stanza does not support probabilities; returning all as 1")

        if isinstance(texts, Iterable) and not isinstance(texts, str):
            docs = [Document([], text=text) for text in texts]
            self.model(docs)
            langs = [self.label_to_lang(doc.lang) for doc in docs]
            probs = [prob] * len(docs)
            return langs if not with_prob else list(zip(langs, probs))
        else:
            raise TypeError(f"Provided {type(texts)=}; must be iterable (list).")
