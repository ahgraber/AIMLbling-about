# %%
import copy
from itertools import chain
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from IPython.display import display
from pydantic import BaseModel
from tqdm.auto import tqdm

# NOTE: RAGAS uses langchain as primary integration, so use it for convenience
from langchain_core.documents import Document as LCDoc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.executor import run_async_batch
from ragas.llms import LangchainLLMWrapper
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
import tiktoken

import pandas as pd

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "ragas-experiment"

DATA_DIR = LOCAL_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
sys.path.insert(0, str(LOCAL_DIR))
from src.ragas.extractors import (  # NOQA: E402
    MarkdownHeadlinesExtractor,
    MarkdownTitleExtractor,
    SpaCyEntities,
    SpacyNERExtractor,
)
from src.utils import check_torch_device, hugo_title_to_h1  # NOQA: E402

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger("src").setLevel(logging.DEBUG)

# %%
_ = load_dotenv()

LLM_KWARGS = {
    "temperature": 0.7,
    "max_tokens": 2048,  # max output tokens
}
RESILIENCE_KWARGS = {
    "max_retries": 10,
    "timeout": 120,
}
device = check_torch_device()

# %% [markdown]
# ## Load files
#
# Source corpus will be blog posts and markdown files from my homelab/gitops repo

# %%
blog_files = list((REPO_DIR / "content" / "blog").rglob("*.md"))
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
kg = KnowledgeGraph(nodes=nodes)

# %% [markdown]
# Use local LMstudio models KG metadata extraction so that the KG is independent of testset / testset generative process

# %%
# # LMStudio provides openai-compatible endpoints
# # get available model names
# client = openai.OpenAI(
#     base_url=os.environ["_LOCAL_BASE_URL"],
#     api_key=os.environ["_LOCAL_API_KEY"],
# )
# print([m.id for m in client.models.list()])
# del client
#
# llm_id="phi-3.1-mini-128k-instruct"
# llm_id="llama-3.2-3b-instruct"
# llm_id="meta-llama-3.1-8b-instruct"
# llm_id="gemma-2-9b-instruct-function-calling"
llm_id = "mistral-nemo-instruct-2407"

# %%
nomic_embedding_kwargs = {
    "device": device.type,
    "trust_remote_code": True,
    "tokenizer_kwargs": {"model_max_length": 8192},
    "model_kwargs": {"rotary_scaling_factor": 2},
    "prompts": {
        # ref: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        # ref: https://sbert.net/examples/applications/computing-embeddings/README.html#prompt-templates
        # ref: https://github.com/run-llama/llama_index/blob/67c7e50e782f9ce12e1fd76b4ac3a131a505f19b/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py#L224-L266
        "text": "search_document: ",  # use key "text" to embed documents
        "query": "search_query: ",  # use key "query" to embed queries
        "clustering": "clustering: ",
        "classification": "classification: ",
    },
    # "default_prompt_name": "",
}
# since LangChain defines encode_kwargs.prompt_name per instance, create an instance per task
cluster_em = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
    model_kwargs=nomic_embedding_kwargs,
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt_name": "clustering",
    },
)
# document_em = HuggingFaceEmbeddings(..., encode_kwargs={..., "prompt_name": "text"})
# query_em = HuggingFaceEmbeddings(..., encode_kwargs={..., "prompt_name": "query"})


# %%
rc = RunConfig(
    timeout=240,
    max_retries=10,
    max_wait=120,
    max_workers=2,
)
llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url=os.environ["_LOCAL_BASE_URL"],
        api_key=os.environ["_LOCAL_API_KEY"],
        model=llm_id,
        **LLM_KWARGS,
        **RESILIENCE_KWARGS,
    ),
    run_config=rc,
)
# em = LangchainEmbeddingsWrapper(
#     OpenAIEmbeddings(
#         base_url=os.environ["_LOCAL_BASE_URL"],
#         api_key=os.environ["_LOCAL_API_KEY"],
#         model="nomic-embed-text-v1.5",
#         check_embedding_ctx_length=False,  # ref: https://github.com/langchain-ai/langchain/issues/21318
#         **RESILIENCE_KWARGS,
#     ),
#     run_config=rc,
# )
em = LangchainEmbeddingsWrapper(
    cluster_em,
    run_config=rc,
)


# %% [markdown]
# ### Define transformations to populate knowledge graph
#
# Specify the transforms and their order to be applied.
#
# Breaking transformations down into stages may make it more resilient

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

kg.save(DATA_DIR / "ragas_knowledgegraph.json")

# %%
kg = KnowledgeGraph().load(DATA_DIR / "ragas_knowledgegraph.json")

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
    kg.save(DATA_DIR / "ragas_knowledgegraph.json")

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
kg = KnowledgeGraph().load(DATA_DIR / "ragas_knowledgegraph.json")

apply_transforms(kg, stage2)
print("Knowledge Graph stage 2 complete.  Saving...")

kg.save(DATA_DIR / "ragas_knowledgegraph.json")


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
# kg.save(DATA_DIR / "ragas_knowledgegraph.json")

# %% [markdown]
# Finally, we can add relationships defined by embeddign similarities
#
# _Tuning the similarity thresholds is critical for extracting high-quality clusters for granular testset generation_

# %%
cosine_sim_builder = CosineSimilarityBuilder(threshold=0.85)
summary_cosine_sim_builder = SummaryCosineSimilarityBuilder(threshold=0.8)

stage3 = [
    cosine_sim_builder,
    summary_cosine_sim_builder,
]

# %%
kg = KnowledgeGraph().load(DATA_DIR / "ragas_knowledgegraph.json")

# remove prior clustering
kg.relationships = [r for r in kg.relationships if r.type not in ["cosine_similarity", "summary_cosine_similarity"]]

apply_transforms(kg, stage3)
print("Knowledge Graph stage 3 complete.  Saving...")

kg.save(DATA_DIR / "ragas_knowledgegraph.json")

# %% [markdown]
# What can we find out about our dataset?

# %%
clusters = kg.find_clusters(relationship_condition=lambda rel: bool(rel.get_property("cosine_similarity")))

print(f"{len(clusters)} found in knowledge graph")


# %%
theme_from_summaries = CommonThemeFromSummariesPrompt()
num_themes = 3
cluster_themes = []
for cluster in tqdm(clusters):
    summaries = Summaries(
        summaries=[node.get_property("summary") for node in cluster if node.get_property("summary") is not None],
        num_themes=num_themes,
    )
    themes = await theme_from_summaries.generate(llm=llm, data=summaries, callbacks=[])  # NOQA: F704
    cluster_themes.append(themes)

for i, themes in enumerate(cluster_themes):
    print("---")
    print(f"Cluster {i} - {len(clusters[i])} nodes with themes:")
    for theme in themes.themes:
        print(theme)

# %%
