import logging
import os
from typing import FrozenSet, Iterable, List, Tuple

from langcodes import Language, standardize_tag
from lingua import Language as LinguaLanguage
import psutil
from typing_extensions import override

from .base import LanguageIDModel

logger = logging.getLogger(__name__)


class Lingua(LanguageIDModel):
    """Classify text with lingua-rs.

    Ref: https://github.com/pemistahl/lingua-rs/tree/main
    """

    name = "lingua"

    def __init__(self, use_small: bool = False):
        p = psutil.Process(os.getpid())
        mem_before = p.memory_info().vms  # .rss

        self.load(use_small)
        _ = self.predict("hello darkness my ol' fren")

        self.memory = (p.memory_info().vms - mem_before) / self._MB
        # TODO: since lingua uses rust crates, does not change python memory allocation

        super().__init__()  # may require model to be loaded

    @override
    def load(self, use_small: bool = False):
        from lingua import LanguageDetectorBuilder

        if use_small:
            self.model = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
        else:
            self.model = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    @override
    def label_to_lang(self, label: LinguaLanguage) -> str:
        """Return standardized tag form of language."""
        lingua2label = label.iso_code_639_1.name
        tag = Language.get(standardize_tag(lingua2label, macro=True)).language
        return tag

    @override
    def _get_labels(self) -> FrozenSet[str]:
        supported_languages = LinguaLanguage.all()
        return frozenset(sorted(self.label_to_lang(label) for label in supported_languages))

    @override
    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        if isinstance(text, str):
            # lingua returns List[ConfidenceValue(language, value)]
            pred = self.model.compute_language_confidence_values(text)[0]
            lang = self.label_to_lang(pred.language)
            prob = pred.value
            return lang if not with_prob else (lang, prob)
        else:
            raise TypeError(f"Provided {type(text)=}; must be string.")

    @override
    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        if isinstance(texts, Iterable) and not isinstance(texts, str):
            # lingua returns List[ConfidenceValue(language, value)]
            preds = self.model.compute_language_confidence_values_in_parallel(texts)
            langs = [self.label_to_lang(pred[0].language) for pred in preds]
            probs = [pred[0].value for pred in preds]
            return langs if not with_prob else list(zip(langs, probs))
        else:
            raise TypeError(f"Provided {type(texts)=}; must be iterable (list).")
