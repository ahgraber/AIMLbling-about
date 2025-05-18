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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "ragas-experiment"

DATA_DIR = LOCAL_DIR / "data"

# %%
sys.path.insert(0, str(LOCAL_DIR))
from src.ragas.helpers import run_ragas_evals, validate_metrics  # NOQA: E402
from src.utils import check_torch_device  # NOQA: E402

# %%
basic_log_config()
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
            timeout=240,
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
for file in sorted(DATA_DIR.glob("ragas_dataset_*.jsonl")):
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
for file in DATA_DIR.glob("qa_response_rag_*.jsonl"):
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
#
# full factorial experiment will be too expensive to run

# %%
# - cull experiment set to only markdown set
experiments = [x for x in experiments if x[0] == "markdown"]
experiment_names = ["_".join(x) for x in experiments]

# - don't need the cross between retrievers, and responders
# (i.e., don't care about who generated the question; want to limit retriever/responder to same provider)
rag_response_df = pd.concat(
    [
        rag_response_df[
            (rag_response_df["experiment"] == f"markdown_{provider}") & (rag_response_df["response_by"] == provider)
        ]
        for provider in providers
    ],
    ignore_index=True,
)
display(rag_response_df)

# %% [markdown]
# #### RAGAS RAG Response Evals
#
# - Faithfulness -
# - Answer/Response Relevance - Is response relevant to the original input?
#   Generate a question based on the response and get the similarity between the original question and generated question


# %%
# convert retrieved_contexts to list of strings
rag_response_df["retrieved_contexts"] = rag_response_df["retrieved_contexts"].apply(
    lambda row: [node["text"] for node in row]
)

# %%
# for experiment, evaluator in itertools.product(experiment_names, providers):
#     # experiment = experiment_names[3]  # "markdown_local"
#     # evaluator = "anthropic"
for experiment in tqdm(experiment_names):
    logger.info(f"Running evals with for {experiment}...")

    source_df = rag_response_df[rag_response_df["experiment"] == experiment].reset_index(drop=True)

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
            if evaluator in ["local", "openai"]  # anthropic, together have high error rates here
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
    # calculate # evals; some evals need > 1 API call
    logger.info(
        f"Evaluations: {len(source_df)=} * {len(response_metrics)=} = {len(source_df) * len(response_metrics):_}"
    )

    run_ragas_evals(
        source_df=source_df,
        metrics=response_metrics,
        outfile=DATA_DIR / f"eval_rag_response_{experiment}.jsonl",
        batch_size=20,
    )

    gc.collect()


# %% [markdown]
# Approx 4 hours per provider (16 hours total)
# Approx token use over eval combinations: 448 questions * 4 experiments * evals
#
# - openai:
#   - in:  2_742_623 (for 1 of 4) // 11_839_585
#   - out: 656_508 (for 1 of 4) // 3_027_981
# - anthropic --> anthropic fails for faithfulness response formatting
#   - in:  5_350_500 (no faithfulness)
#   - out: 323_710 (no faithfulness)
# - together --> llama fails for faithfulness and response relevance response formatting
#   - in: ?
#   - out ?

# %%
e1 = "markdown_together"
e2 = "local"
df = pd.read_json(DATA_DIR / f"eval_rag_response_{e1}-{e2}.jsonl", orient="records", lines=True)

eval_cols = ["semantic_similarity", f"faithfulness_{e2}", f"answer_relevance_{e2}"]
df[eval_cols].isnull().mean()  # % missingness

# %%
