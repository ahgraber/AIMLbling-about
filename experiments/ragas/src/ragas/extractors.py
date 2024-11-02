from collections import defaultdict
import copy
from dataclasses import dataclass, field
from itertools import chain
import logging
import re
import typing as t

from pydantic import BaseModel
from tqdm.auto import tqdm

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRecognizer

from ragas import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.executor import run_async_batch
from ragas.llms import LangchainLLMWrapper
from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import KnowledgeGraph, Node as RagasNode, NodeType
from ragas.testset.synthesizers.prompts import CommonThemeFromSummariesPrompt, Summaries, Theme, Themes
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    Parallel,
    SummaryCosineSimilarityBuilder,
    SummaryExtractor,
    TitleExtractor,
    apply_transforms,
    default_transforms,
)
from ragas.testset.transforms.base import BaseGraphTransformation, Extractor
from ragas.testset.transforms.extractors.llm_based import Headlines
from ragas.testset.transforms.extractors.regex_based import RegexBasedExtractor

logger = logging.getLogger(__name__)


@dataclass
class MarkdownHeadlinesExtractor(Extractor):  # NOQA: D101
    pattern: str = r"^(#{2,3})\s+(.*)"  # look for markdown headings by '#'
    property_name: str = "headlines"

    def _headings_to_headlines(self, headings: t.List[str]) -> Headlines:
        """Naive/hardcoded approach for only ## and ### levels."""
        headlines = {}

        current_section = None
        for level, title in headings:
            _title = f"{level} {title}"
            if level == "##":
                # Create a new section for the main headings
                headlines[_title] = []
                current_section = _title
            elif level == "###" and current_section is not None:
                # Append to the current section if it exists
                headlines[current_section].append(_title)

        return Headlines(headlines=headlines)

    async def extract(self, node: RagasNode) -> t.Tuple[str, t.Any]:
        """Extract headings."""
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise TypeError(f"node.property('page_content') must be a string, found '{type(text)}'")

        matches = re.findall(self.pattern, text, re.MULTILINE)

        if matches:
            headlines = self._headings_to_headlines(headings=matches)
            return self.property_name, headlines.headlines
        else:
            return self.property_name, None


@dataclass
class MarkdownTitleExtractor(Extractor):  # NOQA: D101
    pattern: str = r"^(#{1})\s+(.*)"
    property_name: str = "title"

    async def extract(self, node: RagasNode) -> t.Tuple[str, t.Any]:
        """Extract markdown title."""
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise TypeError(f"node.property('page_content') must be a string, found '{type(text)}'")

        matches = re.findall(self.pattern, text)

        try:
            title = matches[0][1]
        except IndexError:
            return self.property_name, None
        else:
            return self.property_name, title


class SpaCyEntities(BaseModel):
    """Entities from SpaCy NER model."""

    # CARDINAL: t.List[str]=[]
    # DATE: t.List[str]=[]
    # EVENT: t.List[str]=[]
    FAC: t.List[str] = []
    GPE: t.List[str] = []
    LANGUAGE: t.List[str] = []
    LAW: t.List[str] = []
    LOC: t.List[str] = []
    # MONEY: t.List[str]=[]
    NORP: t.List[str] = []
    # ORDINAL: t.List[str]=[]
    ORG: t.List[str] = []
    # PERCENT: t.List[str]=[]
    PERSON: t.List[str] = []
    PRODUCT: t.List[str] = []
    # QUANTITY: t.List[str]=[]
    # TIME: t.List[str]=[]
    WORK_OF_ART: t.List[str] = []


@dataclass
class SpacyNERExtractor(Extractor):  # NOQA: D101
    model: str = "en_core_web_trf"
    parser: Language = field(init=False)
    property_name: str = "entities"

    def __post_init__(self):  # NOQA: D105
        # Check that spaCy model is available; download if not
        if not spacy.util.is_package(self.model):
            spacy.cli.download(self.model)
        self.parser = spacy.load(
            self.model,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
        return super().__post_init__()

    async def extract(self, node: RagasNode) -> t.Tuple[str, t.Any]:
        """Extract named entities with spaCy."""
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise TypeError(f"node.property('page_content') must be a string, found '{type(text)}'")
        parsed = self.parser(text)
        entities = defaultdict(set)
        for ent in parsed.ents:
            entities[ent.label_].add(ent.text)

        return self.property_name, SpaCyEntities(**entities).model_dump()
