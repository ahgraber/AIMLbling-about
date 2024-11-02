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

from IPython.display import Markdown, display
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

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
from src.ragas.helpers import TopKRougeScorer  # NOQA: E402
from src.utils import filter_dict_by_keys  # NOQA:E402

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger("src").setLevel(logging.DEBUG)


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
providers = ["local", "openai", "anthropic", "together"]

# %% [markdown]
# ## Compare retrieved contexts between retrievers with ROUGE
#
# ROUGE --> similarity based on string overlap
#
# We can do this with the vector indexes / retrievers because the text should be similar.<br>
# We _cannot_ do this for the retrieved_context vs ground-truth_context
# because the gt_context is a summary based on the source contexts from the knowledge graph,
# so string overlap doesn't apply


# %%
# Load retrievals from vector indexes

# reduce memory footprint by removing unnecessary info from retrieved nodes during load
filter_node_metadata = partial(filter_dict_by_keys, keys=("node_id", "metadata", "text", "score"))

with (datadir / "rag_retrievals.jsonl").open("r") as f:
    data = [
        {
            k: [filter_node_metadata(node) for node in v] if k in experiment_names else v
            for k, v in json.loads(line).items()
        }
        for line in f
    ]

retrieval_df = pd.DataFrame.from_records(data)
display(retrieval_df)
# 448 rows × 9 columns (testset, query + experiments)

# %%
fname = "rouge_retrieval_similarity.csv"
if (datadir / fname).exists():
    logger.info(f"Prior '{fname}' exists, will not rerun.")
    del fname
else:
    retrieval_scorer = TopKRougeScorer(
        rouge_type="rougeL",
        metric="fmeasure",
        weight=True,
        k=5,
    )

    # extract list of chunk text from node metadata
    for experiment in experiment_names:
        retrieval_df[experiment] = retrieval_df[experiment].apply(lambda row: [node["text"] for node in row])

    # run comparisons
    retrieval_similarities = {}
    for comparison in tqdm(list(itertools.combinations_with_replacement(experiment_names, 2))):
        a, b = comparison
        logger.info(f"{a} vs. {b}")

        retrieval_similarities[comparison] = retrieval_df.apply(
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
    retrieval_similarity_df.to_csv(datadir / fname, index=False)

    del retrieval_df, retrieval_similarity_df, fname

logger.info("ROUGE comparison complete.")

# %% [markdown]
# TODO
# predict score as a function of whether provider was used
#
# - [GPT-4o mini]: https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
# - [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-haiku)
# - [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
# - [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
# - [Artificial Analysis Model Leaderboard](https://artificialanalysis.ai/models/gpt-4o-mini?models_selected=gpt-4o-mini%2Cllama-3-1-instruct-70b%2Cclaude-3-haiku%2Cmistral-nemo)
#
# |                        |  MMLU |  GPQA | HumanEval |  MATH |
# |:-----------------------|------:|------:|----------:|------:|
# | GPT-4o-mini            | 82.0% | 43.0% |     87.2% |   75% |
# | Claude-3-Haiku         | 75.2% | 33.3% |     75.9% |   41% |
# | Llama-3.1-Instruct-70B | 83.6% | 46.7% |     80.5% |   60% |
# | Mistral-Nemo-12b       | 68.0% | 33.0% |     68.0% |   40% |
#
# > Using model cards + Artificial Analysis; higher value if conflicts

# %% [markdown]
# ## Baseline - no RAG
#
# What is "objective" relevance of responses [Semantic Similarity]
# Do models prefer their own responses? [Answer Relevance]

# %%
df = pd.read_json(datadir / "eval_baseline_response.jsonl", orient="records", lines=True)

# %%
# What is "objective" relevance of responses [Semantic Similarity]
semantic_sim = (
    df.groupby("response_by")[["semantic_similarity"]]  # force method chain
    .mean()
    .sort_values(ascending=False, by="semantic_similarity")
)
print(semantic_sim.to_markdown())

# %% [markdown]
# When using "objective" external embedding model (nomic-embed) to assess how relevant (similar) responses are to ground-truth reference,
# models perform as we might expect given their benchmark scores:
# GPT4o-mini, claude-3-haiku, llama-3.1-70b-instruct-turbo (quantized), mistral-nemo-instruct-2407 (12B, quantized)
#
# | response_by | semantic_similarity |
# |:------------|--------------------:|
# | openai      |            0.903384 |
# | anthropic   |            0.89693  |
# | together    |            0.879597 |
# | local       |            0.874995 |
#
# Note: this evaluates at a high level;
# it might be worth doing sentence-level subanalysis to see if the topics are generally correct
# but the fine details differ

# %%
# Do models prefer their own responses? [Answer Relevance]

# set order based on objective performance
order = ["openai", "anthropic", "together", "local"]

response_relevance = (
    df.groupby("response_by")[[f"answer_relevance_{model}" for model in order]]  # force method chain
    .mean()
    .reindex(order)
    # .transpose()
    # .reset_index(names="provider")
)
# re/name columns for clarity
response_relevance.columns = [col.replace("answer_relevance_", "") for col in response_relevance.columns]
response_relevance.columns.name = "evaluation_by"

response_relevance = response_relevance.transpose()
print(response_relevance.to_markdown())

# response_relevance.to_csv(datadir / "baseline_response_relevance.csv")

# %% [markdown]
# Interestingly, claude-3-haiku is scored as having the worst performance,
# and mistral-nemo-instruct-2407 outperforms both claude-3-haiku and llama-3.1-70b (which are much larger models)
#
# Model performance on baseline (no RAG) `answer_relevance` seems more related to model release recency than size,
# with the more recent gpt4o-mini and mistral-nemo-instruct-2407 outperforming older claude-3-haiku and llama-3.1-70b-instruct-turbo
#
# | evaluation_by |   openai |   anthropic |   together |    local |
# |--------------:|---------:|------------:|-----------:|---------:|
# | response_by   |          |             |            |          |
# | openai        | 0.900829 |    0.689515 |   0.851244 | 0.883941 |
# | anthropic     | 0.916623 |    0.732066 |   0.879311 | 0.883678 |
# | together      | 0.91293  |    0.733727 |   0.88017  | 0.882395 |
# | local         | 0.90528  |    0.722939 |   0.870791 | 0.872452 |


# %%
# look at stats per evaluator
evaluator_stats = pd.DataFrame.from_dict(
    {evaluator: df[f"answer_relevance_{evaluator}"].describe().round(4) for evaluator in order}
).transpose()
evaluator_stats.index.name = "evaluation_by"
evaluator_stats = evaluator_stats.transpose()
print(evaluator_stats.to_markdown())

# %%[markdown]
# claude-3-haiku and llama-3.1-70b provide higher/more optimistic mean scores across the board, though gpt-4o-mini has a more optimistic median
#
# | evaluation_by |    openai |   anthropic |   together |     local |
# |:------|----------:|------------:|-----------:|----------:|
# | count | 1792      |   1790      |  1655      | 1792      |
# | mean  |    0.8314 |      0.8529 |     0.8518 |    0.8429 |
# | std   |    0.2753 |      0.2307 |     0.2308 |    0.2334 |
# | min   |    0      |      0      |     0      |    0      |
# | 25%   |    0.8541 |      0.8500 |     0.8502 |    0.8371 |
# | 50%   |    0.9263 |      0.9181 |     0.9126 |    0.9098 |
# | 75%   |    0.9695 |      0.9692 |     0.9675 |    0.9623 |
# | max   |    1      |      1      |     1      |    1      |
#
# All providers show negative skew (mean < median), indicating more scores clustered at higher values.
#
# openai, with the highest variance and greater deviation between mean and median, may demonstrate greater discrimination between "good" and "bad" responses.
#

# %%
metric = "answer_relevance"
for provider in providers:
    pct_missing = df[f"{metric}_{provider}"].isna().sum() / len(df)
    print(f"Evaluation of {metric} with {provider} resulted in {pct_missing:.2%} missingness (evaluation error)")


# %%
