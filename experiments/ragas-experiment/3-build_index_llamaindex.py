# %%
import itertools
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys

from dotenv import load_dotenv
from IPython.display import Markdown, display
from tqdm.auto import tqdm

# use Llamaindex for the rest of the integrations
from llama_index.core import Document as LlamaDoc, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import MarkdownElementNodeParser, MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.duckdb import DuckDBVectorStore
import tiktoken

import pandas as pd

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "ragas-experiment"

DATA_DIR = LOCAL_DIR / "data"

# %%
sys.path.insert(0, str(LOCAL_DIR))
from src.utils import check_torch_device, hugo_title_to_h1  # NOQA: E402

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger("src").setLevel(logging.DEBUG)

# %%
_ = load_dotenv()

LLM_KWARGS = {
    "temperature": 0,
    "max_tokens": 2048,  # max output tokens
}
RESILIENCE_KWARGS = {
    "max_retries": 10,
    "timeout": 60,
}

device = check_torch_device()

# %% [markdown]
# ## Configure service providers via LlamaIndex adapters

# %%
providers = {
    "local": HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
        trust_remote_code=True,
        device=device.type,
        model_kwargs={
            "tokenizer_kwargs": {"model_max_length": 8192},
            "model_kwargs": {"rotary_scaling_factor": 2},
        },
        # ref: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        # ref: https://sbert.net/examples/applications/computing-embeddings/README.html#prompt-templates
        # ref: https://github.com/run-llama/llama_index/blob/67c7e50e782f9ce12e1fd76b4ac3a131a505f19b/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py#L155-L160
        text_instruction="search_document: ",
        query_instruction="search_query: ",
    ),
    "openai": OpenAIEmbedding(
        api_key=os.environ["_OPENAI_API_KEY"],
        model="text-embedding-3-small",
        **RESILIENCE_KWARGS,
    ),
    "anthropic": VoyageEmbedding(
        voyage_api_key=os.environ["_VOYAGE_API_KEY"],
        model_name="voyage-3-lite",
        batch_size=7,  # ref: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/voyageai/
        # **resilience_kwargs,
    ),
    "together": TogetherEmbedding(
        api_base="https://api.together.xyz/v1",
        api_key=os.environ["_TOGETHER_API_KEY"],
        model_name="togethercomputer/m2-bert-80M-8k-retrieval",
        **RESILIENCE_KWARGS,
    ),
}

# %% [markdown]
# ## Load files
#
# Source corpus will be blog posts and markdown files from my homelab/gitops repo


# %%
blog_files = list((REPO_DIR / "content" / "blog").rglob("*.md"))
blog_docs = [
    LlamaDoc(
        text=hugo_title_to_h1(f.read_text()),
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
    itertools.chain(
        (Path.home() / "_code" / "homelab-gitops-k3s").glob("readme.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "docs").rglob("*.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "kubernetes").rglob("*.md"),
    )
)
k3s_docs = [
    LlamaDoc(
        text=f.read_text(),
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
assert len(corpus) == len(documents), "Error: file count in corpus does not match document count"  # NOQA: S101
print(f"{len(documents)=}")

# %%
tokenizer = tiktoken.encoding_for_model("gpt-4")
total_tokens = sum(len(tokenizer.encode(c.read_text())) for c in corpus)
print(f"Corpus contains {total_tokens} tokens")

# %%
md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)

md_nodes = md_parser.get_nodes_from_documents(documents, show_progress=True)

node_tokens = [len(tokenizer.encode(node.text)) for node in md_nodes]
if max(node_tokens) >= 2048:
    logger.warning("Some nodes have token counts >= 2048, which means they may not fit in embedding context")

display(pd.Series(node_tokens).describe())
pd.Series(node_tokens).plot(kind="hist")


# %%
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=0,
)
sentence_nodes = sentence_splitter.get_nodes_from_documents(documents)

node_tokens = [len(tokenizer.encode(node.text)) for node in sentence_nodes]
if max(node_tokens) >= 2048:
    logger.warning("Some nodes have token counts >= 2048, which means they may not fit in embedding context")
display(pd.Series(node_tokens).describe())
pd.Series(node_tokens).plot(kind="hist")


# %%
splitters = {
    "markdown": md_parser,
    "sentence": sentence_splitter,
}

# %% [ markdown]
# ## Define experiments

# %%
experiments = list(
    itertools.product(
        ["markdown", "sentence"],  # chunk experiment
        ["local", "openai", "anthropic", "together"],  # model
    )
)

# %% [markdown]
# ## Build RAG system per experiment
#
# DuckDB uses cosine_similarity internally; this is an exhaustive search

# %%
# params = ("markdown", "local")
for experiment in tqdm(experiments):
    experiment_name = "_".join(experiment)
    logger.info(f"Processing pipeline for {experiment_name} experiment...")

    vector_store = DuckDBVectorStore(f"{experiment_name}.duckdb", persist_dir=str(DATA_DIR / "vectordb"))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Ingest docs directly into a vector db via chunking + embedding pipeline
    chunker, provider = experiment
    pipeline = IngestionPipeline(
        transformations=[
            splitters[chunker],
            providers[provider],
        ],
        # docstore=SimpleDocumentStore(), # docstore not needed since nodes are are persisted in chromadb?
        vector_store=vector_store,
    )
    pipeline.run(documents=documents, show_progress=True)
    pipeline.persist(DATA_DIR / "llamaindex" / f"{experiment_name}_pipeline")


# %% [markdown]
# ## RAG proof of concept

# %%
# load from disk
vector_store = DuckDBVectorStore.from_local(str(DATA_DIR / "vectordb" / f"{experiment_name}.duckdb"))
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=providers[provider],
)

# %%
query = "How can I see what is happening in my Kubernetes cluster?"

# %%
retriever = index.as_retriever(llm=None, similarity_top_k=5)
retriever.retrieve(query)


# %%
from llama_index.llms.openai_like import OpenAILike  # NOQA:E402

# Query Data from the persisted index
query_engine = index.as_query_engine(
    llm=OpenAILike(
        api_base="http://localhost:1234/v1",
        api_key="lmstudio",
        model="mistral-nemo-instruct-2407",
        max_tokens=2048,
    ),
    similarity_top_k=5,
)
response = query_engine.query(query)
display(Markdown(f"<p>{response}</p>"))
