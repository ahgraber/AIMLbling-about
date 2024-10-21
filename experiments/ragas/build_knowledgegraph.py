# %%
from collections import defaultdict
import copy
from dataclasses import dataclass, field
from itertools import chain
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Coroutine, Iterable, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from IPython.display import display
from pydantic import BaseModel
import tiktoken
from tqdm.auto import tqdm

import pandas as pd

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRecognizer

# NOTE: RAGAS uses langchain as primary integration, so use it for convenience
from langchain_core.documents import Document as LCDoc
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from ragas import RunConfig
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import KnowledgeGraph, Node as RagasNode, NodeType
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

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
repo = subprocess.check_output(  # NOQA: S603
    ["git", "rev-parse", "--show-toplevel"],  # NOQA: S607
    cwd=Path(__file__).parent,
    encoding="utf-8",
).strip()
repo = Path(repo).resolve()

datadir = Path(__file__).parent / "data"

# %%
_ = load_dotenv()

# Local uses LMStudio to host a local OpenAI-compatible service
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = "http://localhost:1234/v1"  # os.getenv("LOCAL_BASE_URL")

LLM_KWARGS = {
    "temperature": 0.7,
    "max_tokens": 2048,  # max output tokens
}
RESILIENCE_KWARGS = {
    "max_retries": 10,
    "timeout": 120,
}

# %% [markdown]
# ## Load files
#
# Source corpus will be blog posts and markdown files from my homelab/gitops repo


# %%
def hugo_title_to_h1(text: str):
    """Extract title hugo front matter section and convert to markdown H1."""
    frontmatter = re.match(r"(?P<frontmatter>---[\s\S]*?---)", text)

    try:
        frontmatter = frontmatter["frontmatter"]
    except TypeError:
        # no frontmatter; return without changes
        return text

    title = re.match(r"[\s\S]*title: (?P<title>.*)\n", frontmatter)
    try:
        title = f"# {title['title']}\n\n"  # ensure title has trailing newlines
    except TypeError:
        logger.info("Could not parse title from frontmatter")
        logger.debug({frontmatter["frontmatter"]})
        title = ""
    else:
        text = text.replace(frontmatter, title)
        text = re.sub(r"\n{3,}", "\n\n", text)  # clean up overzealous newlines
        return text


blog_files = list((repo / "content" / "blog").rglob("*.md"))
blog_docs = [
    LCDoc(
        page_content=hugo_title_to_h1(f.read_text()),
        metadata={
            "name": f.parent.name,
            "filename": str(f),
            "category": "blog",
        },
    )
    for f in blog_files
]

# %%
k3s_files = list(
    chain(
        (Path.home() / "_code" / "homelab-gitops-k3s").glob("readme.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "docs").rglob("*.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "kubernetes").rglob("*.md"),
    )
)
k3s_docs = [
    LCDoc(
        page_content=f.read_text(),
        metadata={
            "name": f"{f.parent.name}_{f.name}" if "readme" in f.name.lower() else f.name,
            "filename": str(f),
            "category": "k3s",
        },
    )
    for f in k3s_files
]

# %%
corpus = blog_files + k3s_files
documents = blog_docs + k3s_docs
print(f"{len(documents)=}")

# %% [markdown]
# ## Configure Dataset Generator
#
# Dataset should be independent of any postprocessing (i.e., chunking, vectorizing, reranking, prompt instruction tuning, etc).
# Theoretically, the dataset should also be model-independent, but that's why we're running this experiment.
#
# This means the dataset is preferably generated at the _document_ level, without the influence of the processing pipeline
#
# We can do this with Ragas v0.2 if we construct the knowledge graph separately
# from the procedure of generating the synthetic test set.


# %% [markdown]
# ### Define the transformation pipeline

# %% [markdown]
# Create a headlines extractor for markdown,
# combining markdown Rrgex extractor with nested headlines
#
# This approach is specific to this dataset


# %%
@dataclass
class MarkdownHeadlinesExtractor(Extractor):  # NOQA: D101
    pattern: str = r"^(#{2,3})\s+(.*)"  # look for markdown headings by '#'
    property_name: str = "headlines"

    def _headings_to_headlines(self, headings: List[str]) -> Headlines:
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

    async def extract(self, node: RagasNode) -> Tuple[str, Any]:
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

    async def extract(self, node: RagasNode) -> Tuple[str, Any]:
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


# %% [markdown]
# Create a Named Entity Recognizer using SpaCy


# %%
class SpaCyEntities(BaseModel):
    """Entities from SpaCy NER model."""

    # CARDINAL: List[str]=[]
    # DATE: List[str]=[]
    # EVENT: List[str]=[]
    FAC: List[str] = []
    GPE: List[str] = []
    LANGUAGE: List[str] = []
    LAW: List[str] = []
    LOC: List[str] = []
    # MONEY: List[str]=[]
    NORP: List[str] = []
    # ORDINAL: List[str]=[]
    ORG: List[str] = []
    # PERCENT: List[str]=[]
    PERSON: List[str] = []
    PRODUCT: List[str] = []
    # QUANTITY: List[str]=[]
    # TIME: List[str]=[]
    WORK_OF_ART: List[str] = []


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

    async def extract(self, node: RagasNode) -> Tuple[str, Any]:
        """Extract named entities with spaCy."""
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise TypeError(f"node.property('page_content') must be a string, found '{type(text)}'")
        parsed = self.parser(text)
        entities = defaultdict(set)
        for ent in parsed.ents:
            entities[ent.label_].add(ent.text)

        return self.property_name, SpaCyEntities(**entities).model_dump()


# %% [markdown]
# ### Construct Knowledge Graph


# %%
def lcdoc2rnode(doc: LCDoc) -> RagasNode:
    """Convert LangChain document to Ragas node."""
    return RagasNode(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": doc.page_content,
            "document_metadata": doc.metadata,
        },
    )


nodes = [lcdoc2rnode(doc=doc) for doc in documents]

# %%
kg = KnowledgeGraph(nodes=nodes)

# %% [markdown]
# Use local LMstudio models KG metadata extraction so that the KG is independent of testset / testset generative process
#
# `gemma-2-9b-instruct-function-calling` is _way better_ than llama3.1, llama3.2, or phi3 at returning structurally valid responses!

# %%
# # LMStudio provides openai-compatible endpoints
# # get available model names
# client = openai.OpenAI(
#     base_url=LOCAL_BASE_URL,
#     api_key=LOCAL_API_KEY,
# )
# print([m.id for m in client.models.list()])
# del client

# %%
# model="phi-3.1-mini-128k-instruct"
# model="llama-3.2-3b-instruct"
# model="meta-llama-3.1-8b-instruct"
# model="gemma-2-9b-instruct-function-calling"
model = "mistral-nemo-instruct-2407"

# %%
rc = RunConfig(
    timeout=240,
    max_retries=10,
    max_wait=120,
    max_workers=2,
)
llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY,
        model=model,
        **LLM_KWARGS,
        **RESILIENCE_KWARGS,
    ),
    run_config=rc,
)
em = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY,
        model="nomic-embed-text-v1.5",
        check_embedding_ctx_length=False,  # ref: https://github.com/langchain-ai/langchain/issues/21318
        **RESILIENCE_KWARGS,
    ),
    run_config=rc,
)

# %% [markdown]
# Specify the transforms and their order to be applied.
#
# Breaking transformations down into stages may make it more resilent

# %%
summary_extractor = SummaryExtractor(llm=llm)
# headline_extractor = HeadlinesExtractor(llm=llm)
headline_extractor = MarkdownHeadlinesExtractor()
summary_embedder = EmbeddingExtractor(
    name="summary_embedder",
    property_name="summary_embedding",
    embed_property_name="summary",
    embedding_model=em,
    filter_nodes=lambda node: node.type == NodeType.DOCUMENT,
)


# %%
stage1 = [
    Parallel(
        summary_extractor,
        headline_extractor,
    ),
    summary_embedder,
]
apply_transforms(kg, stage1)
print("Knowledge Graph stage 1 complete.  Saving...")

kg.save(datadir / "ragas_knowledgegraph_v3.json")

# %%
kg = KnowledgeGraph().load(datadir / "ragas_knowledgegraph_v3.json")

# at this stage the knowledge graph should only be at the document level
# so remove/overwrite any other nodes
doc_nodes = [node for node in kg.nodes if node.type == NodeType.DOCUMENT]
assert len(doc_nodes) == len(documents), (  # NOQA: S101
    "Error: document-nodes in knowledge graph do not align with documents from corpus",
)
kg.nodes = doc_nodes

# %% [markdown]
# It is possible some transforms will error out on particular nodes, especially
# if using a local LLM:
#
# ```py
# ValueError: Node 9690569f-d6fc-465f-b077-bc812e8f155b has no summary_embedding
# ```
#
# Manual repair may be necessary


# %%
# identify nodes that failed previously
def missing_summary(node: RagasNode, required_chars: int = 16):
    """Check if summary is."""
    if "summary" in node.properties:
        return (node.get_property("summary") is None) or (len(node.get_property("summary")) < required_chars)


need_summary = [node for node in kg.nodes if missing_summary(node)]
need_headlines = [node for node in kg.nodes if node.get_property("headlines") is None]

print(f"{len(need_summary)=}")
print(f"{len(need_headlines)=}")

# ... repeat for all transforms (especially llm-extraction ones)


# %%
for node in tqdm(need_summary):
    property_name, property_value = await summary_extractor.extract(  # NOQA: F704
        node,
        # # or manually construct if summary is failing
        # RagasNode(properties={'page_content': """..."""})
    )
    # use same test logic to ensure this version will pass
    if missing_summary(RagasNode().add_property(property_name, property_value)):
        node.add_property(property_name, property_value)
    else:
        logger.error(f"Failed to summarize {node}: {node.id}")

need_summary = [node for node in kg.nodes if missing_summary(node)]
print(f"{len(need_summary)=}")

if len(need_summary) == 0:
    kg.save(datadir / "ragas_knowledgegraph_v3.json")

# %% [markdown]
# Now that we have document-level information, we can split the documents into
# smaller chunks and get more specific

# %%
headline_splitter = HeadlineSplitter()
embedding_extractor = EmbeddingExtractor(embedding_model=em)
keyphrase_extractor = KeyphrasesExtractor(llm=llm)
spacy_ner_extractor = SpacyNERExtractor()
# title_extractor = TitleExtractor(llm=llm)
title_extractor = MarkdownTitleExtractor()

stage2 = [
    headline_splitter,
    Parallel(
        summary_extractor,
        embedding_extractor,
        keyphrase_extractor,
        spacy_ner_extractor,
        title_extractor,
    ),
]

# %%
kg = KnowledgeGraph().load(datadir / "ragas_knowledgegraph_v3.json")

apply_transforms(kg, stage2)
print("Knowledge Graph stage 2 complete.  Saving...")

kg.save(datadir / "ragas_knowledgegraph_v3.json")


# %% [markdown]
# After splitting, some nodes may be empty; we should just remove them
# Also, we should validate that these new properties exist for all chunks


# %%
# some nodes are empty; remove
def node_is_empty(node: RagasNode, min_chars: int = 5):
    """Check if node is empty."""
    return len(node.get_property("page_content")) < min_chars


_nodes = copy.deepcopy(kg.nodes)
kg.nodes = [node for node in _nodes if not node_is_empty(node)]

if len(kg.nodes) < len(_nodes):
    logger.info(f"Removed {len(_nodes) - len(kg.nodes)} empty nodes")
assert len(kg.nodes) >= len(corpus), (  # NOQA: S101
    "Error: knowledge graph has fewer nodes than documents from corpus",
)

# %%
need_summary, need_embedding, need_keyphrases, need_entities, need_title = [], [], [], [], []
for node in [n for n in kg.nodes if n.type == NodeType.CHUNK]:
    if missing_summary(node):
        need_summary.append(node)
    if node.get_property("embedding") is None:
        need_embedding.append(node)
    if node.get_property("keyphrases") is None:
        need_keyphrases.append(node)
    if node.get_property("entities") is None:
        need_entities.append(node)
    if node.get_property("title") is None:
        need_title.append(node)

print(f"{len(need_summary)=}")
print(f"{len(need_embedding)=}")
print(f"{len(need_keyphrases)=}")
print(f"{len(need_entities)=}")
print(f"{len(need_title)=}")

# %%
# NOTE: placeholder for repairs as needed given above report
...

# NOTE: don't forget to save!
# kg.save(datadir / "ragas_knowledgegraph_v3.json")

# %% [markdown]
# Finally, we can add relationships defined by embeddign similarities

# %%
cosine_sim_builder = CosineSimilarityBuilder(threshold=0.8)
summary_cosine_sim_builder = SummaryCosineSimilarityBuilder(threshold=0.6)

stage3 = [
    cosine_sim_builder,
    summary_cosine_sim_builder,
]

# %%
kg = KnowledgeGraph().load(datadir / "ragas_knowledgegraph_v3.json")

apply_transforms(kg, stage3)
print("Knowledge Graph stage 3 complete.  Saving...")

kg.save(datadir / "ragas_knowledgegraph_v3.json")

# %%
