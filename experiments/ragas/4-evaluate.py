# %%
import asyncio
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
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm as atqdm
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings

# use Llamaindex for the rest of the integrations
from llama_index.core import Document as LlamaDoc, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
    RougeScore,
    SemanticSimilarity,
)
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
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
from src.llamaindex.prompt_templates import BASELINE_QA_PROMPT, DEFAULT_TEXT_QA_PROMPT  # NOQA: E402
from src.ragas.hacks import TopKRougeScorer  # NOQA: E402
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
for file in datadir.glob("ragas_dataset_*.jsonl"):
    df = pd.read_json(file, orient="records", lines=True)
    df["generated_by"] = file.stem.split("_")[-1]
    dfs.append(df)

# load as pandas df so we can add the retrieved_contexts once they are generated
testset_df = pd.concat(dfs, ignore_index=True)
display(testset_df)
del dfs

# %%
dfs = []
for file in datadir.glob("qa_response_baseline_*.jsonl"):
    df = pd.read_json(file, orient="records", lines=True)
    df["response_by"] = file.stem.split("_")[-1]
    dfs.append(df)

baseline_response_df = pd.concat(dfs, ignore_index=True)
display(baseline_response_df)
del dfs


# %%
# reduce memory footprint by removing unnecessary info from retrieved nodes during load
def filter_node_metadata(d: dict, keys: t.Iterable = ("node_id", "metadata", "text", "score")):
    """Retain only subset of keys."""
    return {k: v for k, v in d.items() if k in keys}


with (datadir / "rag_retrievals.jsonl").open("r") as f:
    data = [
        {
            k: [filter_node_metadata(node) for node in v] if k in experiment_names else v
            for k, v in json.loads(line).items()
        }
        for line in f
    ]

retrieval_df = pd.DataFrame.from_records(data)

# %%
# join testset_df and retrieval_df
experiment_df = pd.concat([testset_df, retrieval_df], axis="columns")

if not all(experiment_df["user_input"] == experiment_df["query"]):
    raise AssertionError("Error: testset_df and retrieval_df are not aligned!")

# reshape into baseline_df format with cols:
# ['user_input', 'reference_contexts', 'reference', 'synthesizer_name', 'generated_by', 'response']
experiment_df = experiment_df.drop(columns="query").melt(
    id_vars=testset_df.columns, value_vars=experiment_names, value_name="retrieved_context", var_name="experiment"
)

display(experiment_df)
# 3584 rows!

# %% [markdown]
# ## Testing plan
#
# 1. For each LLM, run evals _without_ RAG as baseline
# 2. Compare retrievals - how similar are the chunks found by each retriever?
# 3. For each LLM, Run evals with RAG over combination of LLM, EM, and Index


# %%
def validate_metrics(metrics: list):
    """Ensure metrics have been initialized with required model access."""
    for metric in metrics:
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            logger.warning(f"{metric.name} does not have an LLM assigned")
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            logger.warning(f"{metric.name} does not have an embedding model assigned")


def run_ragas_evals(source_df: pd.DataFrame, metrics: list, outfile: str, batch_size: int = 20):
    """Run evaluation of set of metrics for source data."""
    # check if output file exists
    if (datadir / outfile).exists():
        logger.info(f"Prior '{outfile}' exists, will not rerun.")

    # load source data and ensure it has correct features
    missing = required_cols - set(source_df.columns)
    assert not missing, f"Error: source_df missing column(s) {missing}"  # NOQA: S101

    # run evals
    eval_dataset = EvaluationDataset.from_list(source_df.to_dict(orient="records"))
    eval_results = evaluate(dataset=eval_dataset, metrics=metrics, batch_size=batch_size)
    logger.info(f"Summary metrics:\n{eval_results}")

    # save work
    eval_df = pd.concat([source_df, pd.DataFrame.from_records(eval_results.scores)], axis="columns")
    logger.info("Saving eval_results...")
    eval_df.to_json(datadir / outfile, orient="records", lines=True)

    logger.info("Evaluation run complete.")
    gc.collect()


# %% [markdown]
# ### 1. Baseline: How does LLM score without RAG?
#
# Limits to metrics that do not require RAG retrieval:
#
# - Answer/Response Relevance - Is response relevant to the original input?
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

run_ragas_evals(
    source_df=baseline_response_df,
    metrics=baseline_metrics,
    outfile="eval_baseline_response.jsonl",
)

# for provider in tqdm(providers):
#     # check if output file exists
#     if (datadir / f"eval_baseline_{provider}.jsonl").exists():
#         logger.info(f"Prior 'eval_baseline_{provider}.jsonl' exists, will not rerun.")
#         continue

#     # load source data and ensure it has correct features
#     df = pd.read_json(datadir / f"qa_response_baseline_{provider}.jsonl", orient="records", lines=True)
#     missing = required_cols - set(df.columns)
#     assert not missing, f"Error: qa_response_baseline_{provider} missing column(s) {missing}"  # NOQA: S101

#     # run evals
#     eval_dataset = EvaluationDataset.from_list(df.to_dict(orient="records"))
#     eval_results = evaluate(dataset=eval_dataset, metrics=baseline_metrics, batch_size=20)
#     logger.info(f"Summary metrics for {provider}:\n{eval_results}")

#     # save work
#     eval_df = pd.concat([df, pd.DataFrame.from_records(eval_results.scores)], axis="columns")
#     logger.info(f"Saving baseline eval_results for {provider}.")
#     eval_df.to_json(datadir / f"eval_baseline_{provider}.jsonl", orient="records", lines=True)

# logger.info("Baseline response evals complete.")
del baseline_metrics
gc.collect()

# %% [markdown]
# Approx token use over eval combinations
# Note: switch to Haiku due to daily token limit!!
#
# - openai:
#   - in: [4_699_746] @1_540_557
#   - out [179_880] @61_430
# - anthropic
#   - in: [5_294_684] @1_737_885
#   - out [225_121] @76_431
# - together
#   - in: [?]
#   - out [?]

# %% [markdown]
# ### 2. Evaluate retrievals

# %% [markdown]
# #### Check similarity in retrievals
#
# Since source documents are the same, we compare using chunk text; ROUGE-L should work great
#
# Retrieval based on Together embeddings seem to perform poorly / differently than retrieval based on embeddings from other providers

# %%
# pull the retrieval scores from the retrieved chunks
relevance_df = retrieval_df.copy()
for experiment in experiment_names:
    relevance_df[experiment] = relevance_df[experiment].apply(
        lambda row: np.array([node["score"] for node in row]).mean()
    )

display(relevance_df[experiment_names].describe())
display(relevance_df[experiment_names].mean())
# MarkdownNodeParser outperforms SentenceSplitter@512

del relevance_df

# %%
# markdown_local        0.673358
# markdown_openai       0.505389
# markdown_anthropic    0.535638
# markdown_together     0.636722
# sentence_local        0.664001
# sentence_openai       0.481732
# sentence_anthropic    0.507325
# sentence_together     0.612881


# %%
if (datadir / "rouge_retrieval_similarity.csv").exists():
    logger.info("Prior 'rouge_retrieval_similarity.csv' exists, will not rerun.")
else:
    retrieval_scorer = TopKRougeScorer(
        rouge_type="rougeL",
        metric="fmeasure",
        weight=True,
        k=5,
    )

    text_df = retrieval_df.copy()
    for experiment in experiment_names:
        text_df[experiment] = text_df[experiment].apply(lambda row: [node["text"] for node in row])

    # run comparisons
    retrieval_similarities = {}
    for comparison in tqdm(list(itertools.combinations_with_replacement(experiment_names, 2))):
        a, b = comparison
        logger.info(f"{a} vs. {b}")

        retrieval_similarities[comparison] = text_df.apply(
            lambda row: retrieval_scorer.score_lists(row[a], row[b]),  # NOQA: B023 # TODO: FIXME
            axis="columns",
        ).tolist()

    # convert to df
    retrieval_similarities = {k: np.array(v).mean() for k, v in retrieval_similarities.items()}
    retrieval_similarity_df = pd.DataFrame()
    for k, v in retrieval_similarities.items():
        row, col = k
        retrieval_similarity_df.loc[col, row] = v
    display(retrieval_similarity_df)
    retrieval_similarity_df.to_csv(datadir / "rouge_retrieval_similarity.csv", index=False)

    del text_df, retrieval_similarity_df


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

run_ragas_evals(
    source_df=experiment_df,
    metrics=retrieval_metrics,
    outfile="eval_rag_retrieval.jsonl",
)

# for provider in tqdm(providers):
#     # check if output file exists
#     if (datadir / f"eval_retrieval_{provider}.jsonl").exists():
#         logger.info(f"Prior 'eval_retrieval_{provider}.jsonl' exists, will not rerun.")
#         continue

#     # load source data and ensure it has correct features
#     df = experiment_df.copy()
#     missing = required_cols - set(df.columns)
#     assert not missing, f"Error: experiment_df missing column(s) {missing}"  # NOQA: S101

#     # run evals
#     eval_dataset = EvaluationDataset.from_list(df.to_dict(orient="records"))
#     eval_results = evaluate(dataset=eval_dataset, metrics=retrieval_metrics, batch_size=20)
#     logger.info(f"Summary metrics for {provider}:\n{eval_results}")

#     # save work
#     eval_df = pd.concat([df, pd.DataFrame.from_records(eval_results.scores)], axis="columns")
#     logger.info(f"Saving RAGAS retrieval eval_results for {provider}.")
#     eval_df.to_json(datadir / f"eval_retrieval_{provider}.jsonl", orient="records", lines=True)

# logger.info("Retrieval evals complete.")
del retrieval_metrics
gc.collect()

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

run_ragas_evals(
    source_df=experiment_df,
    metrics=response_metrics,
    outfile="eval_rag_response.jsonl",
)

# for provider in tqdm(providers):
#     # check if output file exists
#     fname = f"eval_rag_{provider}.jsonl"
#     if (datadir / fname).exists():
#         logger.info(f"Prior '{fname}' exists, will not rerun.")
#         continue

#     # load source data and ensure it has correct features
#     df = experiment_df.copy()
#     missing = required_cols - set(df.columns)
#     assert not missing, f"Error: experiment_df missing column(s) {missing}"  # NOQA: S101

#     # run evals
#     eval_dataset = EvaluationDataset.from_list(df.to_dict(orient="records"))
#     eval_results = evaluate(dataset=eval_dataset, metrics=response_metrics, batch_size=20)
#     logger.info(f"Summary metrics for {provider}:\n{eval_results}")

#     # save work
#     eval_df = pd.concat([df, pd.DataFrame.from_records(eval_results.scores)], axis="columns")
#     logger.info(f"Saving RAGAS response eval_results for {provider}.")
#     eval_df.to_json(datadir / fname, orient="records", lines=True)

# logger.info("RAG response evals complete.")
del response_metrics
gc.collect()
