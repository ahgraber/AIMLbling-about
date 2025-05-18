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
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.metrics._context_precision import LLMContextPrecisionWithReference

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)
logging.getLogger("ragas.llms").setLevel(logging.ERROR)

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "ragas-experiment"

DATA_DIR = LOCAL_DIR / "data"

# %%
sys.path.insert(0, str(LOCAL_DIR))
from src.ragas.helpers import run_ragas_evals, validate_metrics  # NOQA: E402
from src.utils import check_torch_device, filter_dict_by_keys  # NOQA: E402

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
# Load retrievals from vector indexes

# reduce memory footprint by removing unnecessary info from retrieved nodes during load
filter_node_metadata = partial(filter_dict_by_keys, keys=("node_id", "metadata", "text", "score"))

with (DATA_DIR / "rag_retrievals.jsonl").open("r") as f:
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
if (DATA_DIR / "retriever_relevance.csv").exists():
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

    relevance_df.groupby("experiment")["mean_retrieved_relevance"].describe().to_csv(DATA_DIR / fname)
    display(relevance_df[["experiment", "mean_retrieved_relevance"]].groupby("experiment").mean())
    # MarkdownNodeParser outperforms SentenceSplitter@512

    del relevance_df, fname

# %% [markdown]
# | experiment         |   count |     mean |       std |      min |      25% |      50% |      75% |      max |
# |:-------------------|--------:|---------:|----------:|---------:|---------:|---------:|---------:|---------:|
# | markdown_anthropic |     448 | 0.535638 | 0.0896168 | 0.24064  | 0.479275 | 0.535012 | 0.598882 | 0.781517 |
# | markdown_local     |     448 | 0.673358 | 0.0563549 | 0.536583 | 0.633552 | 0.671418 | 0.714183 | 0.834795 |
# | markdown_openai    |     448 | 0.505379 | 0.097424  | 0.178659 | 0.440267 | 0.504686 | 0.578345 | 0.765099 |
# | markdown_together  |     448 | 0.636722 | 0.116947  | 0.328021 | 0.55386  | 0.6473   | 0.724252 | 0.886935 |
# | sentence_anthropic |     448 | 0.507325 | 0.0891611 | 0.234116 | 0.458913 | 0.508267 | 0.559096 | 0.790512 |
# | sentence_local     |     448 | 0.664001 | 0.0566214 | 0.535727 | 0.624531 | 0.662381 | 0.704818 | 0.826115 |
# | sentence_openai    |     448 | 0.481724 | 0.0985733 | 0.169161 | 0.412585 | 0.482945 | 0.553362 | 0.761959 |
# | sentence_together  |     448 | 0.612881 | 0.117678  | 0.296082 | 0.528364 | 0.623194 | 0.702294 | 0.859675 |
#
# MarkdownNodeParser outperforms SentenceSplitter@512

# %% [markdown]
# ### 3. RAG eval
#
# full factorial experiment will be too expensive to run

# %%
# cull experiment set to only markdown set
experiments = [x for x in experiments if x[0] == "markdown"]
experiment_names = ["_".join(x) for x in experiments]
retrieval_df = retrieval_df[retrieval_df["experiment"].isin(experiment_names)]

# %% [markdown]
# #### RAGAS Retrieval Evals
#
# - Context Precision - Was the retrieved context "useful at arriving at the" in the ground truth _reference_?
#   ContextPrecisionWithoutReference - Was the retrieved context "useful at arriving at the" in the _response_? --> This has strong overlap with Context Recall
# - Context Recall - Can sentences in _reference _ be attributed to the retrieved context?

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


# %%
# run experiments for 'markdown' chunker
for experiment in tqdm(experiment_names):
    logger.info(f"Running retrieval analysis for {experiment}")
    source_df = retrieval_df[retrieval_df["experiment"] == experiment]
    run_ragas_evals(
        source_df=source_df,
        metrics=retrieval_metrics,
        outfile=DATA_DIR / f"eval_rag_retrieval_{experiment}.jsonl",
    )

    gc.collect()

# TODO: redo with together only, merge files

# %% [markdown]
# Estimated approx 2.75 hours per experiment
# Approx token use over eval combinations
#
# - openai:
#   - in:  15_921_001
#   - out: 863_889
# - anthropic
#   - in:  20_196_082
#   - out: 1_292_914
# - together
#   - in: ?
#   - out ?
