# %%
import asyncio
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import typing as t
import warnings

from dotenv import load_dotenv
from IPython.display import Markdown, display
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm as atqdm
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings

# use Llamaindex for the rest of the integrations
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.duckdb import DuckDBVectorStore

# %%
repo = subprocess.check_output(  # NOQA: S603
    ["git", "rev-parse", "--show-toplevel"],  # NOQA: S607
    cwd=Path(__file__).parent,
    encoding="utf-8",
).strip()
repo = Path(repo).resolve()

datadir = Path(__file__).parent / "data"

# %%
sys.path.insert(0, str(Path(__file__).parent))
from src.utils import check_torch_device  # NOQA: E402

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
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

# %%
providers = {
    "local": {
        # "llm": OpenAILike(
        #     api_base=os.environ["_LOCAL_BASE_URL"],
        #     api_key=os.environ["_LOCAL_API_KEY"],
        #     model="mistral-nemo-instruct-2407",
        #     **LLM_KWARGS,
        #     max_retries=10,
        #     timeout=120,
        # ),
        "em": HuggingFaceEmbedding(
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
    },
    "openai": {
        # "llm": OpenAI(
        #     api_key=os.environ["_OPENAI_API_KEY"],
        #     model="gpt-4o-mini",
        #     **LLM_KWARGS,
        #     **RESILIENCE_KWARGS,
        # ),
        "em": OpenAIEmbedding(
            api_key=os.environ["_OPENAI_API_KEY"],
            model="text-embedding-3-small",
            # deployment="text-embedding-3-small",
            **RESILIENCE_KWARGS,
        ),
    },
    "anthropic": {
        # "llm": Anthropic(
        #     api_key=os.environ["_ANTHROPIC_API_KEY"],
        #     model="claude-3-haiku-20240307",
        #     # model="claude-3-5-sonnet-20240620",
        #     **LLM_KWARGS,
        #     **RESILIENCE_KWARGS,
        # ),
        "em": VoyageEmbedding(
            voyage_api_key=os.environ["_VOYAGE_API_KEY"],
            model_name="voyage-3-lite",
            batch_size=7,  # ref: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/voyageai/
            # **resilience_kwargs,
        ),
    },
    "together": {
        # "llm": OpenAILike(
        #     api_base="https://api.together.xyz/v1",
        #     api_key=os.environ["_TOGETHER_API_KEY"],
        #     model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        #     # model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        #     **LLM_KWARGS,
        #     **RESILIENCE_KWARGS,
        # ),
        "em": TogetherEmbedding(
            api_base="https://api.together.xyz/v1",
            api_key=os.environ["_TOGETHER_API_KEY"],
            model_name="togethercomputer/m2-bert-80M-8k-retrieval",
            **RESILIENCE_KWARGS,
        ),
    },
}

# Ragas has not implemented `is_finished` for LlamaIndex models
warnings.filterwarnings("ignore", message="is_finished not implemented for LlamaIndexLLMWrapper")

# %%
# instantiate default "independent" embedding model for evals
default_em = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
    model_kwargs={
        "device": device.type,
        "trust_remote_code": True,
        "tokenizer_kwargs": {"model_max_length": 8192},
        "model_kwargs": {"rotary_scaling_factor": 2},
        "prompts": {
            "text": "search_document: ",
            "query": "search_query: ",
            "clustering": "clustering: ",
            "classification": "classification: ",
        },
        # "default_prompt_name": "",
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt_name": "clustering",
    },
)

# %% [ markdown]
# ## Define experiments

# %%
experiments = list(
    itertools.product(
        ["markdown", "sentence"],  # chunk experiment
        ["local", "openai", "anthropic", "together"],  # model
    )
)
experiment_names = ["_".join(experiment) for experiment in experiments]

# %% [markdown]
# ## Load data

# %%
dfs = []
for file in sorted(datadir.glob("ragas_dataset_*.jsonl")):
    df = pd.read_json(file, orient="records", lines=True)
    df["generated_by"] = file.stem.split("_")[-1]
    dfs.append(df)

# load as pandas df so we can add the retrieved_contexts once they are generated
testset_df = pd.concat(dfs, ignore_index=True)

display(testset_df)
del dfs

# %%[markdown]
# ## Run retrieval for all testset queries

# %%
if (datadir / "rag_retrievals.jsonl").exists():
    logger.info("rag_retrievals.jsonl exists, skipping evaluation...")
else:
    logger.info("Running retrievals across all vector indexes...")

    top_k = 5
    queries = testset_df["user_input"].to_list()
    retrievals = {"query": queries}

    for experiment in tqdm(experiments):
        experiment_name = "_".join(experiment)
        logger.info(f"Retrieving contexts for {experiment_name} experiment...")

        chunker, provider = experiment

        # load vector store from disk
        vector_store = DuckDBVectorStore.from_local(str(datadir / "vectordb" / f"{experiment_name}.duckdb"))
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=providers[provider]["em"],
        )

        # init retriever
        retriever = index.as_retriever(llm=None, similarity_top_k=top_k)

        # run retrievals
        # TODO: async? https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_async/#query-pipeline-with-asyncparallel-execution
        retrievals[experiment_name] = [retriever.retrieve(query) for query in tqdm(queries, leave=False)]

    retrieval_df = pd.DataFrame.from_dict(retrievals)
    retrieval_df.to_json(datadir / "rag_retrievals.jsonl", orient="records", lines=True)
    display(retrieval_df.head())
    del retrievals

    assert all(testset_df["user_input"] == retrieval_df["query"]), "Error: testset_df and retrieval_df are not aligned"  # NOQA SIM104
    logger.info("Retrievals complete.")

# %%
