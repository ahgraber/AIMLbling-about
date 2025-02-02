import logging
from typing import FrozenSet, List, Tuple

from langcodes import Language, standardize_tag

logger = logging.getLogger(__name__)


class LanguageIDModel:
    """Base class for Language Identification Models."""

    name: str
    _MB: int = 1024 * 1024

    def __init__(self):
        self.supported_languages = self._get_labels()

    def load(self):
        """Load model."""
        raise NotImplementedError

    def label_to_lang(self, label: str) -> str:
        """Return standardized tag form of language."""
        tag = Language.get(standardize_tag(label, macro=True)).language
        return tag

    def _get_labels(self) -> FrozenSet[str]:
        """Get valid alpha3 language codes for languages provided by model."""
        raise NotImplementedError

    def predict(self, text: str, with_prob: bool = False) -> str | Tuple[str, float]:
        """Predict the language of a given string."""
        raise NotImplementedError

    def predict_batch(self, texts: List[str], with_prob: bool = False) -> List[str] | List[Tuple[str, float]]:
        """Predict the language of a all strings in list."""
        raise NotImplementedError
