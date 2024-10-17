# %%
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

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
from ragas.testset.transforms import *
from ragas.testset.transforms.base import Extractor

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
# spacy.cli.download("en_core_web_sm")
# spacy.cli.download("en_core_web_trf")

# %%
_ = load_dotenv()

# Local uses LMStudio to host a local OpenAI-compatible service
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")

LLM_KWARGS = {
    "temperature": 0,
    "max_tokens": 2049,  # max output tokens
}
RESILIENCE_KWARGS = {
    "max_retries": 10,
    "timeout": 60,
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
#
# This is essentially the default transformation pipeline from ragas
# but with the SpaCyNERExtractor added

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
    parser: Language = spacy.load(
        "en_core_web_trf",
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    property_name: str = "entities"

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


# nodes = []
# for doc in documents:
#     node = RagasNode(
#         type=NodeType.DOCUMENT,
#         properties={
#             "page_content": doc.page_content,
#             "document_metadata": doc.metadata,
#         },
#     )
#     nodes.append(node)

# spacy_ner_extractor = SpacyNERExtractor()
# [await spacy_ner_extractor.extract(n) for n in nodes]

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
# LMStudio provides openai-compatible endpoints
# get available model names
client = openai.OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
)
print([m.id for m in client.models.list()])
del client

# %%
rc = RunConfig(
    timeout=240,
    max_retries=10,
    max_wait=120,
    max_workers=4,
)
kb_llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url=LOCAL_BASE_URL,
        # base_url="http://localhost:1234/v1",
        api_key=LOCAL_API_KEY,
        # model="meta-llama-3.1-8b-instruct",
        # model="meta-llama-3-8b-instruct-function-calling-json-mode",
        # model="llama-3.2-3b-instruct",
        model="gemma-2-9b-instruct-function-calling",
        **LLM_KWARGS,
        **RESILIENCE_KWARGS,
    ),
    run_config=rc,
)
kb_em = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url=LOCAL_BASE_URL,
        # base_url="http://localhost:1234/v1",
        api_key=LOCAL_API_KEY,
        model="nomic-embed-text-v1.5",
        check_embedding_ctx_length=False,  # ref: https://github.com/langchain-ai/langchain/issues/21318
        **RESILIENCE_KWARGS,
    ),
    run_config=rc,
)

# %% [markdown]
# Specify the transforms and their order to be applied
# Breaking transformations down into stages may make it more resilent

# %%
summary_extractor = SummaryExtractor(llm=kb_llm)
headline_extractor = HeadlinesExtractor(llm=kb_llm)
summary_embedder = EmbeddingExtractor(
    name="summary_embedder",
    property_name="summary_embedding",
    embed_property_name="summary",
    embedding_model=kb_em,
    filter_nodes=lambda node: node.type == NodeType.DOCUMENT,
)

# %%
# TODO: test transforms for response quality/format
# # print(list(enumerate(documents)))
# testcases=[0,1,2,3,12,15,53,103]

# [await headline_extractor.extract(node=nodes[idx]) for idx in testcases]
# [await summary_extractor.extract(node=nodes[idx]) for idx in testcases]

# %%
stage1 = [
    Parallel(
        summary_extractor,
        headline_extractor,
    ),
    summary_embedder,
]
apply_transforms(kg, stage1)  # from ragas.testset.transforms import apply_transforms
print("Knowledge Graph stage 1 complete!")

kg.save(datadir / "ragas_knowledgegraph.json")

# %% [markdown]
# It is possible some transforms will error out on particular nodes:
#
# ```py
# ValueError: Node 9690569f-d6fc-465f-b077-bc812e8f155b has no summary_embedding
# ```
#
# Manual repair may be necessary

# %%
kg = KnowledgeGraph().load(datadir / "ragas_knowledgegraph.json")

# at this stage the knowledge graph should only be at the document level
# so remove/overwrite any other nodes
kg_nodes = [node for node in kg.nodes if node.type == NodeType.DOCUMENT]
assert len(kg_nodes) == len(documents), (  # NOQA: S101
    "Error: document-nodes in knowledge graph do not align with documents from corpus",
)
kg.nodes = kg_nodes

# %%
# identify nodes that failed previously
need_summary = []
need_headlines = []
for node in kg.nodes:
    if node.type == NodeType.DOCUMENT:
        if node.get_property("summary") is None:
            need_summary.append(node)

        if node.get_property("headlines") is None:
            need_headlines.append(node)

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
    node.add_property(property_name, property_value)


# %%
headline_splitter = HeadlineSplitter()
embedding_extractor = EmbeddingExtractor(embedding_model=kb_em)
keyphrase_extractor = KeyphrasesExtractor(llm=kb_llm)
spacy_ner_extractor = SpacyNERExtractor()
title_extractor = TitleExtractor(llm=kb_llm)

# [await keyphrase_extractor.extract(node=nodes[idx]) for idx in testcases]
# [await title_extractor.extract(node=nodes[idx]) for idx in testcases]

stage2 = [
    headline_splitter,
    Parallel(
        embedding_extractor,
        keyphrase_extractor,
        spacy_ner_extractor,
        title_extractor,
    ),
]
apply_transforms(kg, stage2)  # from ragas.testset.transforms import apply_transforms
print("Knowledge Graph stage 2 complete!")

kg.save(datadir / "ragas_knowledgegraph.json")


# %%
cosine_sim_builder = CosineSimilarityBuilder(threshold=0.8)
summary_cosine_sim_builder = SummaryCosineSimilarityBuilder(threshold=0.6)

stage3 = [
    cosine_sim_builder,
    summary_cosine_sim_builder,
]

# %%
apply_transforms(kg, stage3)  # from ragas.testset.transforms import apply_transforms
print("Knowledge Graph stage 3 complete!")

kg.save(datadir / "ragas_knowledgegraph.json")
