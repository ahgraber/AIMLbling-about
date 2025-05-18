import logging
import os
from typing import FrozenSet, Iterable, List, Tuple

import psutil
from typing_extensions import override

from .base import LanguageIDModel

logger = logging.getLogger(__name__)


class Papluca(LanguageIDModel):
    """Classify text with papluca/xlm-roberta-base-language-detection.

    Ref: https://huggingface.co/papluca/xlm-roberta-base-language-detection
    """

    name: str = "papluca/roberta"

    def __init__(self):
        p = psutil.Process(os.getpid())
        mem_before = p.memory_info().vms  # .rss

        self.load()
        _ = self.predict("hello darkness my ol' fren")

        self.memory = (p.memory_info().vms - mem_before) / self._MB

        super().__init__()  # may require model to be loaded

    @override
    def load(self):
        from transformers import pipeline

        from aiml.utils import torch_device

        self.model = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=torch_device(),
        )

    @override
    def _get_labels(self) -> FrozenSet[str]:
        supported_languages = list(self.model.model.config.label2id.keys())
        return frozenset(sorted(self.label_to_lang(label) for label in supported_languages))

    @override
    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        if isinstance(text, str):
            # Model returns List[{'label': str, 'score': float}]
            pred = self.model(text, top_k=1, truncation=True)[0]
            lang = self.label_to_lang(pred["label"])
            prob = pred["score"]
            return lang if not with_prob else (lang, prob)
        else:
            raise TypeError(f"Provided {type(text)=}; must be string.")

    @override
    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        if isinstance(texts, Iterable) and not isinstance(texts, str):
            preds = self.model(texts, top_k=1, truncation=True)
            langs = [self.label_to_lang(pred[0]["label"]) for pred in preds]
            probs = [pred[0]["score"] for pred in preds]
            return langs if not with_prob else list(zip(langs, probs))
        else:
            raise TypeError(f"Provided {type(texts)=}; must be iterable (list).")
