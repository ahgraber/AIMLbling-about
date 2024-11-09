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
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
providers = ["openai", "anthropic", "together", "local"]

# %%
experiments = list(
    itertools.product(
        ["markdown", "sentence"],  # chunk experiment
        providers,
    )
)
experiment_names = ["_".join(experiment) for experiment in experiments]

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

    # display(retrieval_similarity_df)
    # retrieval_similarity_df.to_csv(datadir / fname, index=False)
    md = [e for e in experiment_names if e.startswith("markdown")]
    print(retrieval_similarity_df.loc[md, md].to_markdown())
    s = [e for e in experiment_names if e.startswith("sentence")]
    print(retrieval_similarity_df.loc[s, s].to_markdown())

    del retrieval_df, retrieval_similarity_df, fname

logger.info("ROUGE comparison complete.")

# %%[markdown]
# **markdown splitter experiments**
#
# |                    |   markdown_local |   markdown_openai |   markdown_anthropic |   markdown_together |
# |:-------------------|-----------------:|------------------:|---------------------:|--------------------:|
# | markdown_local     |         0.492455 |        nan        |          nan         |          nan        |
# | markdown_openai    |         0.272449 |          0.490983 |          nan         |          nan        |
# | markdown_anthropic |         0.260526 |          0.296987 |            0.497758  |          nan        |
# | markdown_together  |         0.103195 |          0.105764 |            0.0995554 |            0.455773 |
#
# **sentence splitter experiments**
#
# |                    |   sentence_local |   sentence_openai |   sentence_anthropic |   sentence_together |
# |:-------------------|-----------------:|------------------:|---------------------:|--------------------:|
# | sentence_local     |         0.488204 |        nan        |           nan        |          nan        |
# | sentence_openai    |         0.26766  |          0.485222 |           nan        |          nan        |
# | sentence_anthropic |         0.258991 |          0.286302 |             0.490415 |          nan        |
# | sentence_together  |         0.114583 |          0.112679 |             0.109805 |            0.458982 |
#
# Retrievers self-retrieve very well, but do identify different chunks.
# Nomic, OpenAI, and Voyage all seem to perform similarly, but Together's 8k BERT model severely underperforms

# %% [markdown]
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
baseline_df = pd.read_json(datadir / "eval_baseline_response.jsonl", orient="records", lines=True)

# %%
# What is "objective" relevance of responses [Semantic Similarity]
semantic_sim = (
    baseline_df.groupby("response_by")[["semantic_similarity"]]  # force method chain
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
metrics = [f"answer_relevance_{model}" for model in providers]
response_relevance = (
    baseline_df.groupby("response_by")[metrics]  # force method chain
    .mean()
    .reindex(providers)
)
# re/name columns for clarity
response_relevance.columns = [col.replace("answer_relevance_", "ar_") for col in response_relevance.columns]
response_relevance.columns.name = "evaluation_by"

print(response_relevance.to_markdown())

# response_relevance.to_csv(datadir / "baseline_response_relevance.csv")

# %% [markdown]
# Interestingly, claude-3-haiku is scored as having the worst performance,
# and mistral-nemo-instruct-2407 outperforms both claude-3-haiku and llama-3.1-70b (which are much larger models)
#
# Model performance on baseline (no RAG) `answer_relevance` seems more related to model release recency than size,
# with the more recent gpt4o-mini and mistral-nemo-instruct-2407 outperforming older claude-3-haiku and llama-3.1-70b-instruct-turbo
#
# | response_by   |   ar_openai |   ar_anthropic |   ar_together |   ar_local |
# |:--------------|------------:|---------------:|--------------:|-----------:|
# | openai        |    0.900829 |       0.916623 |      0.91293  |   0.90528  |
# | anthropic     |    0.689515 |       0.732066 |      0.733727 |   0.722939 |
# | together      |    0.851244 |       0.879311 |      0.88017  |   0.870791 |
# | local         |    0.883941 |       0.883678 |      0.882395 |   0.872452 |


# %%
# look at stats per evaluator
evaluator_stats = pd.DataFrame.from_dict(
    {evaluator: baseline_df[f"answer_relevance_{evaluator}"].describe().round(4) for evaluator in providers}
).transpose()
evaluator_stats.index.name = "evaluation_by"
print(evaluator_stats.to_markdown())

# %%[markdown]
# claude-3-haiku and llama-3.1-70b provide higher/more optimistic mean scores across the board, though gpt-4o-mini has a more optimistic median
#
# | evaluation_by   |   count |   mean |    std |   min |    25% |    50% |    75% |   max |
# |:----------------|--------:|-------:|-------:|------:|-------:|-------:|-------:|------:|
# | openai          |    1792 | 0.8314 | 0.2753 |     0 | 0.8541 | 0.9263 | 0.9695 |     1 |
# | anthropic       |    1790 | 0.8529 | 0.2307 |     0 | 0.85   | 0.9181 | 0.9692 |     1 |
# | together        |    1655 | 0.8518 | 0.2308 |     0 | 0.8502 | 0.9126 | 0.9675 |     1 |
# | local           |    1792 | 0.8429 | 0.2334 |     0 | 0.8371 | 0.9098 | 0.9623 |     1 |
#
# All providers show negative skew (mean < median), indicating more scores clustered at higher values.
#
# openai, with the highest variance and greater deviation between mean and median, may demonstrate greater discrimination between "good" and "bad" responses.
#

# %%
# missingness
response_missingness = (
    pd.DataFrame.from_records(
        [
            {"response_by": group[0], **(group[1][metrics].isna().sum() / len(group[1])).to_dict()}
            for group in baseline_df.groupby("response_by")
        ]
    )
    .set_index("response_by")
    .reindex(providers)
)

# re/name columns for clarity
response_missingness.columns = [col.replace("answer_relevance_", "ar_") for col in response_missingness.columns]
response_missingness.columns.name = "evaluation_by"

print(response_missingness.to_markdown())


# %%[markdown]
# | response_by   |   ar_openai |   ar_anthropic |   ar_together |   ar_local |
# |:--------------|------------:|---------------:|--------------:|-----------:|
# | openai        |           0 |     0          |     0.0669643 |          0 |
# | anthropic     |           0 |     0          |     0.0625    |          0 |
# | together      |           0 |     0          |     0.118304  |          0 |
# | local         |           0 |     0.00446429 |     0.0580357 |          0 |


# %%
# TODO
# predict score as a function of whether provider was used

# %%
df = (
    (
        baseline_df[
            [
                "generated_by",
                "response_by",
                # "semantic_similarity",
                *[f"answer_relevance_{provider}" for provider in providers],
            ]
        ]
    )
    .copy()
    .melt(
        id_vars=["generated_by", "response_by"],
        value_vars=[f"answer_relevance_{provider}" for provider in providers],
        var_name="eval_by",
        value_name="answer_relevance",
    )
    .assign(
        eval_by=lambda _df: _df["eval_by"].str.replace("answer_relevance_", ""),
    )
    .dropna()
)
display(df)

# %%

model = smf.ols(
    formula="answer_relevance ~ C(response_by, levels=providers) * C(eval_by, levels=providers)",
    data=df[["answer_relevance", "response_by", "eval_by"]],
)
results = model.fit()
print(results.summary())

# %% [markdown]
# ```txt
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:       answer_relevance   R-squared:                       0.094
# Model:                            OLS   Adj. R-squared:                  0.092
# Method:                 Least Squares   F-statistic:                     48.59
# Date:                Fri, 08 Nov 2024   Prob (F-statistic):          3.36e-138
# Time:                        15:58:47   Log-Likelihood:                 299.98
# No. Observations:                7029   AIC:                            -568.0
# Df Residuals:                    7013   BIC:                            -458.2
# Df Model:                          15
# Covariance Type:            nonrobust
# ===========================================================================================================================================================
#                                                                                               coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                                                   0.9008      0.011     82.141      0.000       0.879       0.922
# C(response_by, levels=providers)[T.anthropic]                                              -0.2113      0.016    -13.625      0.000      -0.242      -0.181
# C(response_by, levels=providers)[T.together]                                               -0.0496      0.016     -3.197      0.001      -0.080      -0.019
# C(response_by, levels=providers)[T.local]                                                  -0.0169      0.016     -1.089      0.276      -0.047       0.014
# C(eval_by, levels=providers)[T.anthropic]                                                   0.0158      0.016      1.018      0.309      -0.015       0.046
# C(eval_by, levels=providers)[T.together]                                                    0.0121      0.016      0.767      0.443      -0.019       0.043
# C(eval_by, levels=providers)[T.local]                                                       0.0045      0.016      0.287      0.774      -0.026       0.035
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.anthropic]     0.0268      0.022      1.220      0.223      -0.016       0.070
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.anthropic]      0.0123      0.022      0.560      0.576      -0.031       0.055
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.anthropic]        -0.0161      0.022     -0.732      0.464      -0.059       0.027
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.together]      0.0321      0.022      1.439      0.150      -0.012       0.076
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.together]       0.0168      0.022      0.748      0.454      -0.027       0.061
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.together]         -0.0136      0.022     -0.612      0.541      -0.057       0.030
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.local]         0.0290      0.022      1.321      0.187      -0.014       0.072
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.local]          0.0151      0.022      0.688      0.491      -0.028       0.058
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.local]            -0.0159      0.022     -0.727      0.467      -0.059       0.027
# ==============================================================================
# Omnibus:                     3092.683   Durbin-Watson:                   1.623
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            13270.352
# Skew:                          -2.193   Prob(JB):                         0.00
# Kurtosis:                       8.106   Cond. No.                         22.8
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# ```

# %%
print(results.summary().as_html())

# %% [markdown]
# <table class="simpletable">
# <caption>OLS Regression Results</caption>
# <tr>
#   <th>Dep. Variable:</th>    <td>answer_relevance</td> <th>  R-squared:         </th> <td>   0.094</td>
# </tr>
# <tr>
#   <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.092</td>
# </tr>
# <tr>
#   <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   48.59</td>
# </tr>
# <tr>
#   <th>Date:</th>             <td>Fri, 08 Nov 2024</td> <th>  Prob (F-statistic):</th> <td>3.36e-138</td>
# </tr>
# <tr>
#   <th>Time:</th>                 <td>16:00:03</td>     <th>  Log-Likelihood:    </th> <td>  299.98</td>
# </tr>
# <tr>
#   <th>No. Observations:</th>      <td>  7029</td>      <th>  AIC:               </th> <td>  -568.0</td>
# </tr>
# <tr>
#   <th>Df Residuals:</th>          <td>  7013</td>      <th>  BIC:               </th> <td>  -458.2</td>
# </tr>
# <tr>
#   <th>Df Model:</th>              <td>    15</td>      <th>                     </th>     <td> </td>
# </tr>
# <tr>
#   <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>
# </tr>
# </table>
# <table class="simpletable">
# <tr>
#                                              <td></td>                                                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
# </tr>
# <tr>
#   <th>Intercept</th>                                                                               <td>    0.9008</td> <td>    0.011</td> <td>   82.141</td> <td> 0.000</td> <td>    0.879</td> <td>    0.922</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]</th>                                           <td>   -0.2113</td> <td>    0.016</td> <td>  -13.625</td> <td> 0.000</td> <td>   -0.242</td> <td>   -0.181</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]</th>                                            <td>   -0.0496</td> <td>    0.016</td> <td>   -3.197</td> <td> 0.001</td> <td>   -0.080</td> <td>   -0.019</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]</th>                                               <td>   -0.0169</td> <td>    0.016</td> <td>   -1.089</td> <td> 0.276</td> <td>   -0.047</td> <td>    0.014</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.anthropic]</th>                                               <td>    0.0158</td> <td>    0.016</td> <td>    1.018</td> <td> 0.309</td> <td>   -0.015</td> <td>    0.046</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.together]</th>                                                <td>    0.0121</td> <td>    0.016</td> <td>    0.767</td> <td> 0.443</td> <td>   -0.019</td> <td>    0.043</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.local]</th>                                                   <td>    0.0045</td> <td>    0.016</td> <td>    0.287</td> <td> 0.774</td> <td>   -0.026</td> <td>    0.035</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.anthropic]</th> <td>    0.0268</td> <td>    0.022</td> <td>    1.220</td> <td> 0.223</td> <td>   -0.016</td> <td>    0.070</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.anthropic]</th>  <td>    0.0123</td> <td>    0.022</td> <td>    0.560</td> <td> 0.576</td> <td>   -0.031</td> <td>    0.055</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.anthropic]</th>     <td>   -0.0161</td> <td>    0.022</td> <td>   -0.732</td> <td> 0.464</td> <td>   -0.059</td> <td>    0.027</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.together]</th>  <td>    0.0321</td> <td>    0.022</td> <td>    1.439</td> <td> 0.150</td> <td>   -0.012</td> <td>    0.076</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.together]</th>   <td>    0.0168</td> <td>    0.022</td> <td>    0.748</td> <td> 0.454</td> <td>   -0.027</td> <td>    0.061</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.together]</th>      <td>   -0.0136</td> <td>    0.022</td> <td>   -0.612</td> <td> 0.541</td> <td>   -0.057</td> <td>    0.030</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.local]</th>     <td>    0.0290</td> <td>    0.022</td> <td>    1.321</td> <td> 0.187</td> <td>   -0.014</td> <td>    0.072</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.local]</th>      <td>    0.0151</td> <td>    0.022</td> <td>    0.688</td> <td> 0.491</td> <td>   -0.028</td> <td>    0.058</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.local]</th>         <td>   -0.0159</td> <td>    0.022</td> <td>   -0.727</td> <td> 0.467</td> <td>   -0.059</td> <td>    0.027</td>
# </tr>
# </table>
# <table class="simpletable">
# <tr>
#   <th>Omnibus:</th>       <td>3092.683</td> <th>  Durbin-Watson:     </th> <td>   1.623</td>
# </tr>
# <tr>
#   <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>13270.352</td>
# </tr>
# <tr>
#   <th>Skew:</th>           <td>-2.193</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
# </tr>
# <tr>
#   <th>Kurtosis:</th>       <td> 8.106</td>  <th>  Cond. No.          </th> <td>    22.8</td>
# </tr>
# </table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# %% [markdown]
# ## RAG Retrieval
#
# Context Precision, Context Recall

# %%
dfs = []
for file in datadir.glob("eval_rag_retrieval_*.jsonl"):
    dfs.append(pd.read_json(file, orient="records", lines=True))

retrieval_metrics_df = pd.concat(dfs, ignore_index=True)
retrieval_metrics_df = retrieval_metrics_df.rename(columns={"experiment": "retriever"})
retrieval_metrics_df.head()

# %%
metrics = [
    *[f"context_precision_{provider}" for provider in providers],
    *[f"context_recall_{provider}" for provider in providers],
]
metrics = [m for m in metrics if m in retrieval_metrics_df.columns]

experiments = [experiment for experiment in experiment_names if experiment.startswith("markdown")]

# %%
# missingness
retrieval_missingness = (
    pd.DataFrame.from_records(
        [
            {"retriever": group[0], **(group[1][metrics].isna().sum() / len(group[1])).to_dict()}
            for group in retrieval_metrics_df.groupby("retriever")
        ]
    )
    .set_index("retriever")
    .reindex(experiments)
)

# %%
print(retrieval_missingness[[col for col in metrics if "precision" in col]].to_markdown())

# %% [markdown]
# | experiment         |   context_precision_openai |   context_precision_anthropic |   context_precision_together |   context_precision_local |
# |:-------------------|---------------------------:|------------------------------:|-----------------------------:|--------------------------:|
# | markdown_openai    |                          0 |                    0.0200893  |                     0.957589 |                         0 |
# | markdown_anthropic |                          0 |                    0.00892857 |                     0.544643 |                         0 |
# | markdown_together  |                          0 |                    0.0334821  |                     0.620536 |                         0 |
# | markdown_local     |                          0 |                    0.0111607  |                     0.455357 |                         0 |


# %%
print(retrieval_missingness[[col for col in metrics if "recall" in col]].to_markdown())

# %% [markdown]
# | experiment         |   context_recall_openai |   context_recall_anthropic |   context_recall_together |   context_recall_local |
# |:-------------------|------------------------:|---------------------------:|--------------------------:|-----------------------:|
# | markdown_openai    |                       0 |                  0.0647321 |                 0.919643  |              0.0267857 |
# | markdown_anthropic |                       0 |                  0.0491071 |                 0.0446429 |              0.0290179 |
# | markdown_together  |                       0 |                  0.0290179 |                 0.0513393 |              0.0178571 |
# | markdown_local     |                       0 |                  0.0602679 |                 0.0200893 |              0.0267857 |


# %%
# metrics
retrieval_metrics = (
    pd.DataFrame.from_records(
        [
            {"retriever": group[0], **group[1][metrics].mean().to_dict()}
            for group in retrieval_metrics_df.groupby("retriever")
        ]
    )
    .set_index("retriever")
    .reindex(experiments)
)

# %%
print(retrieval_metrics[[col for col in metrics if "precision" in col]].to_markdown())

# %% [markdown]
# | experiment         |   context_precision_openai |   context_precision_anthropic |   context_precision_together |   context_precision_local |
# |:-------------------|---------------------------:|------------------------------:|-----------------------------:|--------------------------:|
# | markdown_openai    |                   0.697613 |                      0.825519 |                     0.629094 |                  0.539757 |
# | markdown_anthropic |                   0.709266 |                      0.845001 |                     0.684157 |                  0.559164 |
# | markdown_together  |                   0.233777 |                      0.374917 |                     0.183431 |                  0.157313 |
# | markdown_local     |                   0.673546 |                      0.793435 |                     0.604087 |                  0.523723 |


# %%
print(retrieval_metrics[[col for col in metrics if "recall" in col]].to_markdown())

# %% [markdown]
# | experiment         |   context_recall_openai |   context_recall_anthropic |   context_recall_together |   context_recall_local |
# |:-------------------|------------------------:|---------------------------:|--------------------------:|-----------------------:|
# | markdown_openai    |                0.80878  |                   0.921943 |                  0.639815 |               0.728984 |
# | markdown_anthropic |                0.81638  |                   0.926123 |                  0.704505 |               0.791533 |
# | markdown_together  |                0.354486 |                   0.52335  |                  0.267961 |               0.379621 |
# | markdown_local     |                0.751547 |                   0.867002 |                  0.610267 |               0.703121 |


# %% [markdown]
# ## RAG Response

# %%
dfs = []
for file in datadir.glob("eval_rag_response_*.jsonl"):
    dfs.append(pd.read_json(file, orient="records", lines=True))

response_metrics_df = pd.concat(dfs, ignore_index=True)
display(response_metrics_df.head())
del dfs

# %%
metrics = [  # manually ordered
    "semantic_similarity",
    *[f"answer_relevance_{provider}" for provider in providers],
    *[f"faithfulness_{provider}" for provider in providers],
]
metrics = [m for m in metrics if m in response_metrics_df.columns]

experiments = [experiment for experiment in experiment_names if experiment.startswith("markdown")]


# %%
# missingness
response_missingness = (
    pd.DataFrame.from_records(
        [
            {"experiment": group[0], **(group[1][metrics].isna().sum() / len(group[1])).to_dict()}
            for group in response_metrics_df.groupby("experiment")
        ]
    )
    .set_index("experiment")
    .reindex(experiments)
)

# %%
print(response_missingness[["semantic_similarity"]].to_markdown())

# %%[markdown]
# | experiment         |   semantic_similarity |
# |:-------------------|----------------------:|
# | markdown_openai    |                     0 |
# | markdown_anthropic |                     0 |
# | markdown_together  |                     0 |
# | markdown_local     |                     0 |

# %%
print(response_missingness[[col for col in metrics if "faithfulness" in col]].to_markdown())

# %%[markdown]
# | experiment         |   faithfulness_openai |   faithfulness_local |
# |:-------------------|----------------------:|---------------------:|
# | markdown_openai    |             0.0848214 |            0.0825893 |
# | markdown_anthropic |             0.0290179 |            0.0200893 |
# | markdown_together  |             0.0803571 |            0.111607  |
# | markdown_local     |             0.0446429 |            0.0625    |

# %%
print(response_missingness[[col for col in metrics if "answer_relevance" in col]].to_markdown())

# %%[markdown]
# | experiment         |   answer_relevance_openai |   answer_relevance_anthropic |   answer_relevance_together |   answer_relevance_local |
# |:-------------------|--------------------------:|-----------------------------:|----------------------------:|-------------------------:|
# | markdown_openai    |                         0 |                            0 |                    0.102679 |               0.0267857  |
# | markdown_anthropic |                         0 |                            0 |                    0.109375 |               0          |
# | markdown_together  |                         0 |                            0 |                    0.245536 |               0          |
# | markdown_local     |                         0 |                            0 |                    0.133929 |               0.00223214 |

# %%
# metrics
response_metrics = (
    pd.DataFrame.from_records(
        [
            {"response_by": group[0], **group[1][metrics].mean().to_dict()}
            for group in response_metrics_df.groupby("response_by")
        ]
    )
    .set_index("response_by")
    .reindex(providers)
)

# %%
print(response_metrics[["semantic_similarity"]].to_markdown())

# %%[markdown]
# | experiment         |   semantic_similarity |
# |:-------------------|----------------------:|
# | markdown_openai    |              0.927751 |
# | markdown_anthropic |              0.916914 |
# | markdown_together  |              0.88384  |
# | markdown_local     |              0.906305 |

# %%
print(response_metrics[[col for col in metrics if "faithfulness" in col]].to_markdown())

# %%[markdown]
# | experiment         |   faithfulness_openai |   faithfulness_local |
# |:-------------------|----------------------:|---------------------:|
# | markdown_openai    |              0.75366  |             0.701613 |
# | markdown_anthropic |              0.767491 |             0.679363 |
# | markdown_together  |              0.436331 |             0.421242 |
# | markdown_local     |              0.626723 |             0.598187 |

# %%
print(response_metrics[[col for col in metrics if "answer_relevance" in col]].to_markdown())

# %%[markdown]
# | experiment         |   answer_relevance_openai |   answer_relevance_anthropic |   answer_relevance_together |   answer_relevance_local |
# |:-------------------|--------------------------:|-----------------------------:|----------------------------:|-------------------------:|
# | markdown_openai    |                  0.917483 |                     0.924802 |                    0.917282 |                 0.905744 |
# | markdown_anthropic |                  0.864219 |                     0.909169 |                    0.8972   |                 0.868209 |
# | markdown_together  |                  0.747819 |                     0.784185 |                    0.783476 |                 0.777769 |
# | markdown_local     |                  0.852791 |                     0.868579 |                    0.870687 |                 0.858142 |

# %%
# TODO
# predict score as a function of whether provider was used

# %%
df = (
    (
        response_metrics_df[
            [
                "generated_by",
                # "experiment",
                "response_by",
                # "semantic_similarity",
                # "faithfulness_local",
                # "faithfulness_openai",
                *[f"answer_relevance_{provider}" for provider in providers],
            ]
        ]
    )
    .copy()
    # .rename(columns={"experiment": "retriever"})
    .melt(
        id_vars=["generated_by", "response_by"],  # "retriever",
        value_vars=[f"answer_relevance_{provider}" for provider in providers],
        var_name="eval_by",
        value_name="answer_relevance",
    )
    .assign(
        # retriever=lambda _df: _df["retriever"].str.replace("markdown_", ""),
        eval_by=lambda _df: _df["eval_by"].str.replace("answer_relevance_", ""),
    )
    .dropna()
)
display(df)

# %%
model = smf.ols(
    formula="answer_relevance ~ C(response_by, levels=providers) * C(eval_by, levels=providers)",
    data=df[["answer_relevance", "response_by", "eval_by"]],
)
results = model.fit()
print(results.summary())

# %%[markdown]
# ```txt
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:       answer_relevance   R-squared:                       0.068
# Model:                            OLS   Adj. R-squared:                  0.066
# Method:                 Least Squares   F-statistic:                     33.30
# Date:                Fri, 08 Nov 2024   Prob (F-statistic):           2.72e-93
# Time:                        16:14:26   Log-Likelihood:                 1216.7
# No. Observations:                6890   AIC:                            -2401.
# Df Residuals:                    6874   BIC:                            -2292.
# Df Model:                          15
# Covariance Type:            nonrobust
# ===========================================================================================================================================================
#                                                                                               coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                                                   0.9175      0.010     95.645      0.000       0.899       0.936
# C(response_by, levels=providers)[T.anthropic]                                              -0.0533      0.014     -3.926      0.000      -0.080      -0.027
# C(response_by, levels=providers)[T.together]                                               -0.1697      0.014    -12.507      0.000      -0.196      -0.143
# C(response_by, levels=providers)[T.local]                                                  -0.0647      0.014     -4.769      0.000      -0.091      -0.038
# C(eval_by, levels=providers)[T.anthropic]                                                   0.0073      0.014      0.540      0.590      -0.019       0.034
# C(eval_by, levels=providers)[T.together]                                                   -0.0002      0.014     -0.014      0.989      -0.028       0.027
# C(eval_by, levels=providers)[T.local]                                                      -0.0117      0.014     -0.859      0.390      -0.039       0.015
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.anthropic]     0.0376      0.019      1.961      0.050    2.22e-05       0.075
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.anthropic]      0.0290      0.019      1.514      0.130      -0.009       0.067
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.anthropic]         0.0085      0.019      0.441      0.659      -0.029       0.046
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.together]      0.0332      0.020      1.680      0.093      -0.006       0.072
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.together]       0.0359      0.020      1.774      0.076      -0.004       0.075
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.together]          0.0181      0.020      0.913      0.361      -0.021       0.057
# C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.local]         0.0157      0.019      0.817      0.414      -0.022       0.053
# C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.local]          0.0417      0.019      2.166      0.030       0.004       0.079
# C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.local]             0.0171      0.019      0.887      0.375      -0.021       0.055
# ==============================================================================
# Omnibus:                     4417.436   Durbin-Watson:                   1.892
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            40235.917
# Skew:                          -3.092   Prob(JB):                         0.00
# Kurtosis:                      13.096   Cond. No.                         22.6
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# ```

# %%
print(results.summary().as_html())

# %% [markdown]
# <table class="simpletable">
# <caption>OLS Regression Results</caption>
# <tr>
#   <th>Dep. Variable:</th>    <td>answer_relevance</td> <th>  R-squared:         </th> <td>   0.068</td>
# </tr>
# <tr>
#   <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.066</td>
# </tr>
# <tr>
#   <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   33.30</td>
# </tr>
# <tr>
#   <th>Date:</th>             <td>Fri, 08 Nov 2024</td> <th>  Prob (F-statistic):</th> <td>2.72e-93</td>
# </tr>
# <tr>
#   <th>Time:</th>                 <td>16:14:42</td>     <th>  Log-Likelihood:    </th> <td>  1216.7</td>
# </tr>
# <tr>
#   <th>No. Observations:</th>      <td>  6890</td>      <th>  AIC:               </th> <td>  -2401.</td>
# </tr>
# <tr>
#   <th>Df Residuals:</th>          <td>  6874</td>      <th>  BIC:               </th> <td>  -2292.</td>
# </tr>
# <tr>
#   <th>Df Model:</th>              <td>    15</td>      <th>                     </th>     <td> </td>
# </tr>
# <tr>
#   <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>
# </tr>
# </table>
# <table class="simpletable">
# <tr>
#                                              <td></td>                                                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
# </tr>
# <tr>
#   <th>Intercept</th>                                                                               <td>    0.9175</td> <td>    0.010</td> <td>   95.645</td> <td> 0.000</td> <td>    0.899</td> <td>    0.936</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]</th>                                           <td>   -0.0533</td> <td>    0.014</td> <td>   -3.926</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.027</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]</th>                                            <td>   -0.1697</td> <td>    0.014</td> <td>  -12.507</td> <td> 0.000</td> <td>   -0.196</td> <td>   -0.143</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]</th>                                               <td>   -0.0647</td> <td>    0.014</td> <td>   -4.769</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.038</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.anthropic]</th>                                               <td>    0.0073</td> <td>    0.014</td> <td>    0.540</td> <td> 0.590</td> <td>   -0.019</td> <td>    0.034</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.together]</th>                                                <td>   -0.0002</td> <td>    0.014</td> <td>   -0.014</td> <td> 0.989</td> <td>   -0.028</td> <td>    0.027</td>
# </tr>
# <tr>
#   <th>C(eval_by, levels=providers)[T.local]</th>                                                   <td>   -0.0117</td> <td>    0.014</td> <td>   -0.859</td> <td> 0.390</td> <td>   -0.039</td> <td>    0.015</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.anthropic]</th> <td>    0.0376</td> <td>    0.019</td> <td>    1.961</td> <td> 0.050</td> <td> 2.22e-05</td> <td>    0.075</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.anthropic]</th>  <td>    0.0290</td> <td>    0.019</td> <td>    1.514</td> <td> 0.130</td> <td>   -0.009</td> <td>    0.067</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.anthropic]</th>     <td>    0.0085</td> <td>    0.019</td> <td>    0.441</td> <td> 0.659</td> <td>   -0.029</td> <td>    0.046</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.together]</th>  <td>    0.0332</td> <td>    0.020</td> <td>    1.680</td> <td> 0.093</td> <td>   -0.006</td> <td>    0.072</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.together]</th>   <td>    0.0359</td> <td>    0.020</td> <td>    1.774</td> <td> 0.076</td> <td>   -0.004</td> <td>    0.075</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.together]</th>      <td>    0.0181</td> <td>    0.020</td> <td>    0.913</td> <td> 0.361</td> <td>   -0.021</td> <td>    0.057</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.anthropic]:C(eval_by, levels=providers)[T.local]</th>     <td>    0.0157</td> <td>    0.019</td> <td>    0.817</td> <td> 0.414</td> <td>   -0.022</td> <td>    0.053</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.together]:C(eval_by, levels=providers)[T.local]</th>      <td>    0.0417</td> <td>    0.019</td> <td>    2.166</td> <td> 0.030</td> <td>    0.004</td> <td>    0.079</td>
# </tr>
# <tr>
#   <th>C(response_by, levels=providers)[T.local]:C(eval_by, levels=providers)[T.local]</th>         <td>    0.0171</td> <td>    0.019</td> <td>    0.887</td> <td> 0.375</td> <td>   -0.021</td> <td>    0.055</td>
# </tr>
# </table>
# <table class="simpletable">
# <tr>
#   <th>Omnibus:</th>       <td>4417.436</td> <th>  Durbin-Watson:     </th> <td>   1.892</td>
# </tr>
# <tr>
#   <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>40235.917</td>
# </tr>
# <tr>
#   <th>Skew:</th>           <td>-3.092</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
# </tr>
# <tr>
#   <th>Kurtosis:</th>       <td>13.096</td>  <th>  Cond. No.          </th> <td>    22.6</td>
# </tr>
# </table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

#
#
# R-squared and Adjusted R-squared: The R-squared value is 0.068, which means that the independent variables in the model explain about 6.8% of the variance in the dependent variable. The adjusted R-squared, which takes into account the number of predictors, is 0.066, so the model fit is relatively low.
# F-statistic and p-value: The F-statistic is 33.30, with a p-value of 2.72e-93, which indicates that the model as a whole is statistically significant.
# Coefficients: The coefficients represent the change in the dependent variable (answer_relevance) associated with a one-unit change in the independent variable, while holding all other variables constant.
#
# The intercept of 0.9092 represents the expected value of answer_relevance when all other variables are 0.
# The coefficients for the interaction terms (e.g., C(response_by)[T.local]:C(eval_by)[anthropic]) represent the difference in the effect of the "response_by" variable depending on the "eval_by" variable.
# For example, the coefficient of -0.0406 for C(response_by)[T.local]:C(eval_by)[anthropic] means that when the response is by "local" and the evaluation is by "anthropic", the expected answer_relevance is 0.0406 lower compared to the baseline.

# %%
