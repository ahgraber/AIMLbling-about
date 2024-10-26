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
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.async_utils import run_async_tasks
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper, LlamaIndexLLMWrapper

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
from src.utils import batched, check_torch_device  # NOQA: E402

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
            model="claude-3-5-sonnet-20240620",
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
# ## Load testset(s)

# %%
dfs = []
for provider in providers:
    df = pd.read_json(datadir / f"ragas_dataset_{provider}.jsonl", orient="records", lines=True)
    df["provider"] = provider
    dfs.append(df)

# load as pandas df so we can add the retrieved_contexts once they are generated
testset_df = pd.concat(dfs, ignore_index=True)
display(testset_df)
del dfs

# %% [markdown]
# ## Testing plan
#
# 1. Compare retrievals - how similar are the chunks found by each retriever?
# 2. For each LLM, run evals _without_ RAG as baseline
# 3. For each LLM, Run evals with RAG over combination of LLM, EM, and Index

# %% [markdown]
# ## Run retrieval for all testset queries

# %%
if (datadir / "rag_retrievals.jsonl").exists():
    logger.info("Loading retrieval_df from jsonl file...")
    retrieval_df = pd.read_json(datadir / "rag_retrievals.jsonl", orient="records", lines=True)

else:
    logger.info("Running retrievals across all vector indexes...")

    top_k = 5
    queries = testset_df["user_input"].to_list()
    retrievals = {"query": queries}

    for experiment in tqdm(experiments):
        experiment_name = "_".join(experiment)
        logger.info(f"Evaluating retrieval similarity for {experiment_name} experiment...")

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
        # TODO: async: https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_async/#query-pipeline-with-asyncparallel-execution
        retrievals[experiment_name] = [retriever.retrieve(query) for query in tqdm(queries, leave=False)]

    retrieval_df = pd.DataFrame.from_dict(retrievals)
    retrieval_df.to_json(datadir / "rag_retrievals.jsonl", orient="records", lines=True)
    display(retrieval_df.head())
    del retrievals

assert all(testset_df["user_input"] == retrieval_df["query"]), "Error: testset_df and retrieval_df are not aligned"  # NOQA SIM104

# %% [markdown]
# ## Check similarity in retrievals
#
# Since source documents are the same, we compare using chunk text; ROUGE-L should work great
#
# Retrieval based on Together embeddings seem to perform poorly / differently than retrieval based on embeddings from other providers

# %%
relevance_df = retrieval_df.copy()
for experiment in experiment_names:
    relevance_df[experiment] = relevance_df[experiment].apply(
        lambda row: np.array([node["score"] for node in row]).mean()
    )

display(relevance_df[experiment_names].describe())
display(relevance_df[experiment_names].mean())
# MarkdownNodeParser outperforms SentenceSplitter@512

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
class TopKRougeScorer:
    """Calculate similarity between lists of strings."""

    def __init__(
        self,
        rouge_type: str,
        use_stemmer: bool = False,
        split_summaries: bool = False,
        metric: str = "fmeasure",
        weight: bool = True,
        k: int = 5,
    ):
        self.rouge_type = rouge_type
        if metric in ["precision", "recall", "fmeasure"]:
            self.metric = metric
        else:
            raise ValueError("metric not in ['precision','recall','fmeasure']")

        self.scorer = rouge_scorer.RougeScorer(
            [rouge_type],  # use rougeL, which ignores/concats newlines
            use_stemmer=use_stemmer,
            split_summaries=split_summaries,
        )

        self.weights = self._calculate_rank_weights(k) if weight else None

    def _calculate_rank_weights(self, k: int) -> np.ndarray:
        """Create a weight matrix of shape (m, n) with weights diminishing from the identity."""
        # Create arrays of row and column indices
        indices = np.arange(k)

        # Calculate the distance matrix
        distance_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])

        # Create the weight matrix using the distance
        weight_matrix = 1 / (1 + distance_matrix)
        return weight_matrix

    def score(self, x: str, y: str) -> float:
        """Evaluate Rouge score for string pairs and extract the specified metric."""
        return self.scorer.score(x, y)[self.rouge_type]._asdict()[self.metric]

    def score_lists(self, a: t.List[str], b: t.List[str]) -> float:
        """Compare lists of retrieved text using ROUGE."""
        pairs = itertools.product(a, b)
        scores = np.array([self.score(x, y) for x, y in pairs]).reshape(len(a), len(b))

        if self.weights is None:
            return scores.mean()
        else:
            return np.average(scores, weights=self.weights)


# %%
retrieval_scorer = TopKRougeScorer(
    rouge_type="rougeL",
    metric="fmeasure",
    weight=True,
    k=5,
)

text_df = retrieval_df.copy()
for experiment in experiment_names:
    text_df[experiment] = text_df[experiment].apply(lambda row: [node["text"] for node in row])

retrieval_similarities = {}
for comparison in tqdm(list(itertools.combinations_with_replacement(experiment_names, 2))):
    a, b = comparison
    logger.info(f"{a} vs. {b}")

    retrieval_similarities[comparison] = text_df.apply(
        lambda row: retrieval_scorer.score_lists(row[a], row[b]), axis="columns"
    ).tolist()

# %%
retrieval_similarities = {k: np.array(v).mean() for k, v in retrieval_similarities.items()}
retrieval_similarity_df = pd.DataFrame()
for k, v in retrieval_similarities.items():
    row, col = k
    retrieval_similarity_df.loc[col, row] = v
display(retrieval_similarity_df)

# %%
# 	markdown_local	markdown_openai	markdown_anthropic	markdown_together	sentence_local	sentence_openai	sentence_anthropic	sentence_together
# markdown_local	0.492455	NaN	NaN	NaN	NaN	NaN	NaN	NaN
# markdown_openai	0.272337	0.490984	NaN	NaN	NaN	NaN	NaN	NaN
# markdown_anthropic	0.260526	0.296956	0.497758	NaN	NaN	NaN	NaN	NaN
# markdown_together	0.103195	0.105769	0.099555	0.455773	NaN	NaN	NaN	NaN
# sentence_local	0.202283	0.177978	0.168717	0.087573	0.488204	NaN	NaN	NaN
# sentence_openai	0.177671	0.202049	0.177056	0.089394	0.267586	0.485222	NaN	NaN
# sentence_anthropic	0.177539	0.188289	0.199449	0.087041	0.258991	0.286270	0.490415	NaN
# sentence_together	0.090568	0.089018	0.084187	0.142817	0.114583	0.112678	0.109805	0.458982

# %% [markdown]
# ## Evaluate with RAGAS

# %%
from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
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
    is_reproducible,
)

# import nest_asyncio

# nest_asyncio.apply()

# %% [markdown]
# ### Baseline: How does LLM score without RAG?
#
# Limits to metrics that do not require RAG retrieval:
#
# - Answer/Response Relevance - Is response relevant to the original input?
#   Generate a question based on the response and get the similarity between the original question and generated question
# - SemanticSimilarity - similarity between ground truth reference and response


# %%
# Generate A's for all Q's in testset for all providers
for provider in tqdm(providers):
    df = testset_df.copy()

    # don't need to wrap in LlamaIndexLLMWrapper b/c using LlamaIndex directly here
    llm = providers[provider]["llm"]

    tasks = [llm.achat(messages=BASELINE_QA_PROMPT.format_messages(query_str=query)) for query in df["user_input"]]

    responses = [run_async_tasks(tasks=batch, show_progress=True) for batch in tqdm(batched(tasks, n=5), leave=False)]

    df["response"] = [response.message.content for response in itertools.chain.from_iterable(responses)]
    df.to_json(datadir / f"qa_baseline_{provider}.jsonl", orient="records", lines=True)


# %%
metrics = [
    ResponseRelevancy(),
    SemanticSimilarity(),
]
required_cols = set(itertools.chain.from_iterable(metric.required_columns["SINGLE_TURN"] for metric in metrics))

# %%
# TODO
# - set nonstandard openai env vars
# - custom batch run_as_async / Executor.results()


# %%
# for provider in tqdm(providers):
provider = list(providers.keys())[0]
baseline_df = pd.read_json(datadir / f"qa_baseline_{provider}.jsonl", orient="records", lines=True)

metrics = [
    *[
        ResponseRelevancy(name=f"answer_relevancy_{evaluator}", llm=LlamaIndexLLMWrapper(providers[evaluator]["llm"]))
        for evaluator in providers
    ],
    SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(default_em)),
]
missing = required_cols - set(baseline_df.columns)
assert not missing, f"Error: qa_baseline_{provider} missing column(s) {missing}"

eval_dataset = EvaluationDataset.from_list(baseline_df.to_dict(orient="records"))
evals = evaluate(dataset=eval_dataset, metrics=metrics)
logger.info(f"Summary metrics for {provider}:\n{evals}")

eval_df = pd.concat([baseline_df, pd.DataFrame.from_records(evals.scores)], axis="columns")

eval_df.to_json(datadir / f"eval_baseline_{provider}.jsonl", orient="records", lines=True)

# %% [markdown]
# ## RAG eval
#
# - Context Recall - Can sentences in response be attributed to the retrieved context?
# - Context Precision - Was the retrieved context "useful at arriving at the" in the ground truth _reference_?
#   ContextPrecisionWithoutReference - Was the retrieved context "useful at arriving at the" in the _response_? --> This has strong overlap with Context Recall
# - Faithfulness -
# - Answer/Response Relevance - Is response relevant to the original input?
#   Generate a question based on the response and get the similarity between the original question and generated question

# %%
metrics = [
    LLMContextRecall(),  # retrieved_context used in response
    LLMContextPrecisionWithoutReference(),  # retrieved_context relevant to response --> strong overlap w/ ContextRecall
    LLMContextPrecisionWithReference(),  # retrieved_context relevant to reference
    SemanticSimilarity(),
    Faithfulness(),  # response wrt retrieved_context
    ResponseRelevancy(),  # response wrt input
]

# %%
experiment = experiments[0]
# for experiment in experiments:

experiment_name = "_".join(experiment)
chunker, provider = experiment

df = testset_df.copy()
df["retrieved_context"] = retrieval_df[experiment_name]


# %%
# load from disk
vector_store = DuckDBVectorStore.from_local(str(datadir / "vectordb" / f"{experiment_name}.duckdb"))
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=providers[provider],
    use_async=True,  # TODO: does this work?
)

# %%
query = "How can I see what is happening in my Kubernetes cluster?"

# %%
retriever = index.as_retriever(llm=None, similarity_top_k=5)
retriever.retrieve(query)


# %%
# Query Data from the persisted index
query_engine = index.as_query_engine(
    llm=OpenAILike(
        api_base=os.environ["_LOCAL_BASE_URL"],
        api_key=os.environ["_LOCAL_API_KEY"],
        model="mistral-nemo-instruct-2407",
        max_tokens=2048,
    ),
    similarity_top_k=5,
)
response = query_engine.query(query)
display(Markdown(f"<p>{response}</p>"))

# %%
