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
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.metrics._context_precision import LLMContextPrecisionWithReference

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

# %% [markdown]
# ## Load data
#
# - synthetic testset
# - baseline (no retrieval) responses
# - retrievals
# - RAG responses

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
# Load baseline response
dfs = []
for file in sorted(datadir.glob("qa_response_baseline_*.jsonl")):
    df = pd.read_json(file, orient="records", lines=True)
    df["response_by"] = file.stem.split("_")[-1]
    if not all(testset_df["user_input"] == df["user_input"]):
        raise AssertionError(f"Error: testset_df and {file} are not aligned!")
    dfs.append(df)

baseline_response_df = pd.concat(dfs, ignore_index=True)

if not all(
    testset_df["user_input"].tolist() * len(testset_df["generated_by"].unique()) == baseline_response_df["user_input"]
):
    raise AssertionError("Error: testset_df and retrieval_df are not aligned!")

display(baseline_response_df)
# 1792 rows (testset * providers)
del dfs


# %%
# Load retrievals from vector indexes


# reduce memory footprint by removing unnecessary info from retrieved nodes during load
def filter_dict_by_keys(d: dict, keys: t.Iterable):
    """Retain only subset of keys."""
    return {k: v for k, v in d.items() if k in keys}


filter_node_metadata = partial(filter_dict_by_keys, keys=("node_id", "metadata", "text", "score"))

with (datadir / "rag_retrievals.jsonl").open("r") as f:
    data = [
        {
            k: [filter_node_metadata(node) for node in v] if k in experiment_names else v
            for k, v in json.loads(line).items()
        }
        for line in f
    ]

# join testset_df and retrieval data
retrieval_df = pd.concat([testset_df, pd.DataFrame.from_records(data)], axis="columns")

if not all(retrieval_df["user_input"] == retrieval_df["query"]):
    raise AssertionError("Error: testset_df and retrieval_df are not aligned!")

# reshape into baseline_df format with cols:
# ['user_input', 'reference_contexts', 'reference', 'synthesizer_name', 'generated_by', 'response']
retrieval_df = retrieval_df.drop(columns="query").melt(
    id_vars=testset_df.columns, value_vars=experiment_names, value_name="retrieved_contexts", var_name="experiment"
)

display(retrieval_df)
# 3584 rows (testset * experiments)

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
# ## Testing plan
#
# 1. For each LLM, run evals _without_ RAG as baseline
# 2. Compare retrievals - how similar are the chunks found by each retriever?
# 3. For each LLM, Run evals with RAG over combination of LLM, EM, and Index


# %% [markdown]
# ### 1. Baseline: How does LLM score without RAG?
#
# Limits to metrics that do not require RAG retrieval:
#
# - Answer/Response Relevance - Is response relevant to the original input?<br>
#   Generate a question based on the response and get the similarity between the original question and generated question
# - SemanticSimilarity - similarity between ground truth reference and response

# %%
baseline_metrics = [
    *[
        ResponseRelevancy(
            name=f"answer_relevance_{evaluator}",
            llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
            embeddings=LangchainEmbeddingsWrapper(default_em),
        )
        for evaluator in providers
    ],
    SemanticSimilarity(
        llm=LlamaIndexLLMWrapper(providers["local"]["llm"]),  # I don't think this is used
        embeddings=LangchainEmbeddingsWrapper(default_em),
    ),
]
required_cols = set(
    itertools.chain.from_iterable(metric.required_columns["SINGLE_TURN"] for metric in baseline_metrics)
)

validate_metrics(baseline_metrics)
logger.info(f"API calls: {len(baseline_response_df)=} * {len(baseline_metrics)=}")

run_ragas_evals(
    source_df=baseline_response_df,
    metrics=baseline_metrics,
    outfile=datadir / "eval_baseline_response.jsonl",
)

del baseline_metrics
gc.collect()

# %%
# %% [markdown]
# Approx token use over eval combinations
# Note: switch to Haiku due to daily token limit!!
#
# - openai:
#   - in:  4_619_124
#   - out: 179_109
# - anthropic
#   - in:   5_118_460
#   - out:  246_175    @
# - together
#   - in: ?
#   - out ?

# %% [markdown]
# ### 2. Evaluate retrievals

# %% [markdown]
# #### Check similarity in retrievals
#
# Since source documents are the same, we compare using chunk text; ROUGE-L should work great
#
# Retrieval based on Together embeddings seem to perform poorly / differently than retrieval based on embeddings from other providers

# %%
# Retriever self-reported relevance
fname = "retriever_relevance.csv"
if (datadir / "retriever_relevance.csv").exists():
    logger.info(f"Prior '{fname}' exists, will not rerun.")
    del fname
else:
    relevance_df = retrieval_df.copy()
    # pull the retrieval scores from the retrieved chunks
    relevance_df["scores"] = relevance_df["retrieved_contexts"].apply(
        lambda row: np.array([node["score"] for node in row])
    )
    relevance_df["mean_retrieved_relevance"] = relevance_df["scores"].apply(np.mean)
    relevance_df[["experiment", "mean_retrieved_relevance"]].groupby("experiment").mean()

    relevance_df[["experiment", "mean_retrieved_relevance"]].groupby("experiment").describe().to_csv(datadir / fname)
    display(relevance_df[["experiment", "mean_retrieved_relevance"]].groupby("experiment").mean())
    # MarkdownNodeParser outperforms SentenceSplitter@512

    del relevance_df, fname

# %%
# 	mean_retrieved_relevance
# markdown_anthropic	0.535638
# markdown_local	0.673358
# markdown_openai	0.505379
# markdown_together	0.636722
# sentence_anthropic	0.507325
# sentence_local	0.664001
# sentence_openai	0.481724
# sentence_together	0.612881


# %% [markdown]
# ### 3. RAG eval
#
# - Context Precision - Was the retrieved context "useful at arriving at the" in the ground truth _reference_?
#   ContextPrecisionWithoutReference - Was the retrieved context "useful at arriving at the" in the _response_? --> This has strong overlap with Context Recall
# - Context Recall - Can sentences in _reference _ be attributed to the retrieved context?
# - Faithfulness -
# - Answer/Response Relevance - Is response relevant to the original input?
#   Generate a question based on the response and get the similarity between the original question and generated question

# %% [markdown]
# #### RAGAS Retrieval Evals

# %%
retrieval_metrics = [
    *[
        LLMContextPrecisionWithReference(  # retrieved_context relevant to _reference answer_
            name=f"context_precision_{evaluator}",
            llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
        )
        for evaluator in providers
    ],
    # *[
    #     LLMContextPrecisionWithoutReference(  # retrieved_context relevant to _generated response_ --> strong overlap w/ ContextRecall
    #         name=f"context_precision_noref_{evaluator}",
    #         llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
    #     )
    #     for evaluator in providers
    # ],
    *[
        LLMContextRecall(  # retrieved_context used in _reference answer_
            name=f"context_recall_{evaluator}",
            llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]),
        )
        for evaluator in providers
    ],
]
required_cols = set(
    itertools.chain.from_iterable(metric.required_columns["SINGLE_TURN"] for metric in retrieval_metrics)
)

validate_metrics(retrieval_metrics)
logger.info(f"API calls: {len(retrieval_df)=} * {len(retrieval_metrics)=}")

# convert retrieved_contexts to list of strings
retrieval_df["retrieved_contexts"] = retrieval_df["retrieved_contexts"].apply(
    lambda row: [node["text"] for node in row]
)

run_ragas_evals(
    source_df=retrieval_df,
    metrics=retrieval_metrics,
    outfile=datadir / "eval_rag_retrieval.jsonl",
)

del retrieval_metrics
gc.collect()

# %% [markdown]
# Approx token use over eval combinations
# Note: switch to Haiku due to daily token limit!!
#
# - openai:
#   - in:  @4_650_606
#   - out: @180_333
# - anthropic
#   - in:  @8_703_682
#   - out:   @  887_272
# - together
#   - in: ?
#   - out ?

# %% [markdown]
# #### RAGAS Response Evals

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

run_ragas_evals(
    source_df=rag_response_df,
    metrics=response_metrics,
    outfile=datadir / "eval_rag_response.jsonl",
)

del response_metrics
gc.collect()
