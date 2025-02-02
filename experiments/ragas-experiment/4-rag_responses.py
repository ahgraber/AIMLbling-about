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
from ragas.async_utils import run_async_tasks

import numpy as np
import pandas as pd

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
repo = get_repo_path(this_file())

datadir = Path(this_file()).parent / "data"

# %%
sys.path.insert(0, str(Path(this_file()).parent))
from src.llamaindex.prompt_templates import BASELINE_QA_PROMPT, DEFAULT_TEXT_QA_PROMPT  # NOQA: E402
from src.utils import check_torch_device  # NOQA: E402

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
for file in sorted(datadir.glob("ragas_dataset_*.jsonl")):
    df = pd.read_json(file, orient="records", lines=True)
    df["generated_by"] = file.stem.split("_")[-1]
    dfs.append(df)

# load as pandas df so we can add the retrieved_contexts once they are generated
testset_df = pd.concat(dfs, ignore_index=True)

display(testset_df)
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
    id_vars=testset_df.columns,
    value_vars=experiment_names,
    value_name="retrieved_contexts",
    var_name="experiment",
)
del testset_df, retrieval_df

display(experiment_df)
# 3584 rows!

# %%
# ## Generate RAG A's for all Q's in testset for all providers

# %%
for provider in tqdm(providers):
    if (datadir / f"qa_response_rag_{provider}.jsonl").exists():
        logger.info(f"'qa_response_rag_{provider}.jsonl' exists, skipping generation...")
        continue

    logger.info(f"Evaluating responses for {provider}...")

    df = experiment_df.copy()

    # don't need to wrap in LlamaIndexLLMWrapper b/c using LlamaIndex directly here
    llm = providers[provider]["llm"]

    tasks = [
        llm.achat(
            messages=DEFAULT_TEXT_QA_PROMPT.format_messages(
                context_str="\n---\n".join(node["text"] for node in record["retrieved_contexts"]),
                query_str=record["user_input"],
            )
        )
        for record in df[["user_input", "retrieved_contexts"]].to_dict(orient="records")
    ]

    # run async to help reduce timeout errors (especially on local generation)
    # # responses = [run_async_tasks(tasks=batch, show_progress=True) for batch in tqdm(batched(tasks, n=5), leave=False)]
    responses = run_async_tasks(  # requires https://github.com/explodinggradients/ragas/pull/1589
        tasks=tasks, show_progress=True, batch_size=20
    )

    df["response"] = [response.message.content for response in responses]
    df.to_json(datadir / f"qa_response_rag_{provider}.jsonl", orient="records", lines=True)

logger.info("RAG answer generation complete.")


# %% [markdown]
# Approx token use for generation
# Note: switch to Haiku due to daily token limit!!
#
# - openai:
#   - in: 4_905_289
#   - out: 1_201_517
# - anthropic
#   - in: 5_687_881
#   - out: 1_236_788
# - together
#   - in: ?
#   - out ?
