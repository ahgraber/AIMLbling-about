# %%
import asyncio
from functools import partial
import gc
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
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings

# use Llamaindex for the rest of the integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.together import TogetherLLM
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    SemanticSimilarity,
)

import matplotlib.pyplot as plt

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
from src.ragas.helpers import run_ragas_evals, validate_metrics  # NOQA: E402
from src.utils import check_torch_device  # NOQA: E402

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)
logging.getLogger("ragas.llms").setLevel(logging.ERROR)


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
        "llm": OpenAILike(
            api_base=os.environ["_LOCAL_BASE_URL"],
            api_key=os.environ["_LOCAL_API_KEY"],
            model="mistral-nemo-instruct-2407",
            **LLM_KWARGS,
            max_retries=10,
            timeout=120,
        ),
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
        "llm": OpenAI(
            api_key=os.environ["_OPENAI_API_KEY"],
            model="gpt-4o-mini",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": OpenAIEmbedding(
            api_key=os.environ["_OPENAI_API_KEY"],
            model="text-embedding-3-small",
            # deployment="text-embedding-3-small",
            **RESILIENCE_KWARGS,
        ),
    },
    "anthropic": {
        "llm": Anthropic(
            api_key=os.environ["_ANTHROPIC_API_KEY"],
            model="claude-3-haiku-20240307",
            # model="claude-3-5-sonnet-20240620",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": VoyageEmbedding(
            voyage_api_key=os.environ["_VOYAGE_API_KEY"],
            model_name="voyage-3-lite",
            batch_size=7,  # ref: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/voyageai/
            # **resilience_kwargs,
        ),
    },
    "together": {
        "llm": OpenAILike(
            api_base="https://api.together.xyz/v1",
            api_key=os.environ["_TOGETHER_API_KEY"],
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            # model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": TogetherEmbedding(
            api_base="https://api.together.xyz/v1",
            api_key=os.environ["_TOGETHER_API_KEY"],
            model_name="togethercomputer/m2-bert-80M-8k-retrieval",
            **RESILIENCE_KWARGS,
        ),
    },
}


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

# %%
# Load synthetic testset
dfs = []
for file in sorted(datadir.glob("ragas_dataset_*.jsonl")):
    df = pd.read_json(file, orient="records", lines=True)
    df["generated_by"] = file.stem.split("_")[-1]
    dfs.append(df)

# load as pandas df so we can add the retrieved_contexts once they are generated
testset_df = pd.concat(dfs, ignore_index=True)

display(testset_df)
# 448 rows (~100 sample questions /providers)
del dfs

# %%
# load RAG responses
dfs = []
for file in datadir.glob("qa_response_rag_*.jsonl"):
    df = pd.read_json(file, orient="records", lines=True)
    df["response_by"] = file.stem.split("_")[-1]
    dfs.append(df)

rag_response_df = pd.concat(dfs, ignore_index=True)

if not all(
    testset_df["user_input"].tolist() * len(testset_df["generated_by"].unique()) * len(experiments)
    == rag_response_df["user_input"]
):
    raise AssertionError("Error: testset_df and retrieval_df are not aligned!")

display(rag_response_df)
# 14336 rows (testset * providers * experiments)
del dfs

# %% [markdown]
# ### 3. RAG eval

# %%
# cull experiment set to only markdown set
experiments = [x for x in experiments if x[0] == "markdown"]
experiment_names = ["_".join(x) for x in experiments]
retrieval_df = rag_response_df[rag_response_df["experiment"].isin(experiment_names)]

# %% [markdown]
# #### RAGAS RAG Response Evals
#
# - Faithfulness -
# - Answer/Response Relevance - Is response relevant to the original input?
#   Generate a question based on the response and get the similarity between the original question and generated question


# %%
response_metrics = [
    SemanticSimilarity(
        llm=LlamaIndexLLMWrapper(providers["local"]["llm"]),  # I don't think this is used
        embeddings=LangchainEmbeddingsWrapper(default_em),
    ),
    *[
        Faithfulness(  # response wrt retrieved_context
            name=f"faithfulness_{evaluator}",
            llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
        )
        for evaluator in providers
    ],
    *[
        ResponseRelevancy(  # response wrt input
            name=f"answer_relevance_{evaluator}",
            llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
            embeddings=LangchainEmbeddingsWrapper(default_em),
        )
        for evaluator in providers
    ],
]
required_cols = set(
    itertools.chain.from_iterable(metric.required_columns["SINGLE_TURN"] for metric in response_metrics)
)

validate_metrics(response_metrics)
logger.info(f"API calls: {len(rag_response_df)=} * {len(response_metrics)=}")

# %%
for evaluator in providers:
    source_df = rag_response_df[rag_response_df["response_by"] == evaluator]

    run_ragas_evals(
        source_df=source_df,
        metrics=response_metrics,
        outfile=datadir / f"eval_rag_response_{evaluator}.jsonl",
    )

    gc.collect()


# %% [markdown]
# Estimated approx 37 hours
# Approx token use over eval combinations
#
# - openai:
#   - in:  (4_650_606 - ... ) + ...
#   - out: (180_333 - ... ) + ...
# - anthropic
#   - in:  (8_703_682 - ... ) + ...
#   - out: (887_272 - ... ) + ...
# - together
#   - in: ?
#   - out ?
