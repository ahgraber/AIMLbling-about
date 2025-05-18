# %%
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from IPython.display import display
from pydantic import BaseModel
from tqdm.auto import tqdm

# NOTE: RAGAS uses langchain as primary integration, so use it for convenience
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from ragas import RunConfig
from ragas.cost import CostCallbackHandler, TokenUsage, get_token_usage_for_anthropic, get_token_usage_for_openai
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, NodeType
from ragas.testset.synthesizers import (
    AbstractQuerySynthesizer,
    ComparativeAbstractQuerySynthesizer,
    SpecificQuerySynthesizer,
)
import tiktoken

import pandas as pd

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "ragas-experiment"

DATA_DIR = LOCAL_DIR / "data"

# %%
sys.path.insert(0, str(LOCAL_DIR))
from src.ragas.hacks import llama_finished_parser  # NOQA: E402
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

# %% [markdown]
# ## Configure LLMs via LangChain adapters
#
# This allows for easy integration w/ RAGAS

# %%
# ref: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
# ref: https://sbert.net/examples/applications/computing-embeddings/README.html#prompt-templates
# ref: https://github.com/run-llama/llama_index/blob/67c7e50e782f9ce12e1fd76b4ac3a131a505f19b/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py#L224-L266

nomic_embedding_kwargs = {
    "device": device.type,
    "trust_remote_code": True,
    "tokenizer_kwargs": {"model_max_length": 8192},
    "model_kwargs": {"rotary_scaling_factor": 2},
    "prompts": {
        "text": "search_document: ",  # use key "text" to embed documents
        "query": "search_query: ",  # use key "query" to embed queries
        "clustering": "clustering: ",
        "classification": "classification: ",
    },
    # "default_prompt_name": "",
}
# since LangChain defines encode_kwargs.prompt_name per instance, create an instance per task
cluster_em = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
    model_kwargs=nomic_embedding_kwargs,
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt_name": "clustering",
    },
)
# document_em = HuggingFaceEmbeddings(..., encode_kwargs={..., "prompt_name": "text"})
# query_em = HuggingFaceEmbeddings(..., encode_kwargs={..., "prompt_name": "query"})


# %%
providers = {
    "local": {
        "llm": ChatOpenAI(
            base_url=os.environ["_LOCAL_BASE_URL"],
            # base_url="http://localhost:1234/v1",
            api_key=os.environ["_LOCAL_API_KEY"],
            # model="phi-3.1-mini-128k-instruct",
            model="mistral-nemo-instruct-2407",
            **LLM_KWARGS,
            max_retries=10,
            timeout=120,
        ),
        # "em": OpenAIEmbeddings(
        #     base_url=os.environ["_LOCAL_BASE_URL"],
        #     # base_url="http://localhost:1234/v1",
        #     api_key=os.environ["_LOCAL_API_KEY"],
        #     model="nomic-embed-text-v1.5",
        #     check_embedding_ctx_length=False,  # ref: https://github.com/langchain-ai/langchain/issues/21318
        #     **RESILIENCE_KWARGS,
        # ),
        "em": cluster_em,
    },
    "openai": {
        "llm": ChatOpenAI(
            api_key=os.environ["_OPENAI_API_KEY"],
            model="gpt-4o-mini",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": OpenAIEmbeddings(
            api_key=os.environ["_OPENAI_API_KEY"],
            model="text-embedding-3-small",
            # deployment="text-embedding-3-small",
            **RESILIENCE_KWARGS,
        ),
    },
    "anthropic": {
        "llm": ChatAnthropic(
            api_key=os.environ["_ANTHROPIC_API_KEY"],
            # model="claude-3-5-sonnet-20240620",
            model="claude-3-haiku-20240307",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": VoyageAIEmbeddings(
            api_key=os.environ["_VOYAGE_API_KEY"],
            model="voyage-3-lite",
            batch_size=7,  # ref: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/voyageai/
            # **resilience_kwargs,
        ),
    },
    "together": {
        "llm": ChatOpenAI(  # Langchain-Together is based on Langchain-OpenAI; might as well use source
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["_TOGETHER_API_KEY"],
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            # model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            # **LLM_KWARGS,
            **{
                "temperature": 0,
                "max_tokens": 4096,  # max output tokens
            },
            **RESILIENCE_KWARGS,
        ),
        "em": TogetherEmbeddings(
            api_key=os.environ["_TOGETHER_API_KEY"],
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            **RESILIENCE_KWARGS,
        ),
    },
}


# %% [markdown]
# ## Generate the test dataset per model

# %%
# provider = "together"
for provider in tqdm(providers):
    rc = RunConfig(
        timeout=240 if provider == "local" else 120,
        max_retries=10,
        max_wait=120 if provider == "local" else 60,
        max_workers=2 if provider == "local" else 4,
    )
    llm = LangchainLLMWrapper(
        providers[provider]["llm"],
        run_config=rc,
        is_finished_parser=lambda x: True,  # parser is gives false errors for Together
        # is_finished_parser=llama_finished_parser,
    )
    em = LangchainEmbeddingsWrapper(
        providers[provider]["em"],
        run_config=rc,
    )
    # if provider == 'anthropic':
    #     cost_callback = CostCallbackHandler(token_usage_parser=get_token_usage_for_anthropic)
    # elif provider in ['openai','together']:
    #     cost_callback = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    # else:
    #     logger.info("No cost callback handler assigned.")

    # reload kg each time
    kg = KnowledgeGraph().load(DATA_DIR / "ragas_knowledgegraph.json")
    generator = TestsetGenerator(
        llm=llm,
        knowledge_graph=kg,
    )

    # define synthesizers
    abstract_query_synth = AbstractQuerySynthesizer(llm=llm)
    comparative_query_synth = ComparativeAbstractQuerySynthesizer(llm=llm)
    specific_query_synth = SpecificQuerySynthesizer(llm=llm)

    logger.info(f"Generating testset with {provider}...")

    # NOTE: we don't use 'generate_with...docs()' because we already use our pre-created knowledge graph
    dataset = generator.generate(
        testset_size=100,
        query_distribution=[
            (abstract_query_synth, 0.25),
            (comparative_query_synth, 0.25),
            (specific_query_synth, 0.5),
        ],
        # callbacks=[cost_callback],
        # run_config=...,
    )

    dataset.to_jsonl(DATA_DIR / f"ragas_dataset_{provider}.jsonl")

    df = dataset.to_pandas()
    display(df)

    logger.info("Testset generation complete.")

logger.info("All testsets have been created!")

# %% [markdown]
# Approx token use for 100-question dataset generation
#
# - openai:
#   - in: [517_300, 220_500]
#   - out [4_700, 15_500]
# - anthropic
#   - in: [605_200, 257_200]
#   - out [20_300, 20_600]
# - together
#   - in: [?]
#   - out [?]


# %% [markdown]
# ## Cost Estimation

# # %%
# # ref: https://docs.ragas.io/en/stable/howtos/applications/_cost/#understanding-tokenusageparser

# tokenizer = tiktoken.encoding_for_model("gpt-4")
# total_tokens = sum(len(tokenizer.encode(c.read_text())) for c in corpus)
# print(f"Corpus contains {total_tokens} tokens")

# print("Embedding:")
# emb_pricing = {
#     "text-embedding-3-small": {
#         "input": 0.02,
#     },
#     "text-embedding-3-large": {
#         "input": 0.13,
#     },
#     "voyage-3": {  # anthropic partner
#         "input": 0.06,
#     },
#     "voyage-3-lite": {  # anthropic partner
#         "input": 0.02,
#     },
#     "M2-BERT-80M-8K-Retrieval": {  # together.ai
#         "input": 0.008,
#     },
# }
# for model, price in emb_pricing.items():
#     # this is an overestimate b/c I should use nowhere near this number of output tokens
#     input_cost = price["input"] * total_tokens / 1_000_000
#     cost = input_cost
#     print(f"  {model} will cost ${cost:,.4f}")

# # Ref: https://huggingface.co/spaces/philschmid/llm-pricing
# # PPM -> price-per-million
# print("Generation:")
# llm_pricing = {
#     "gpt-4o": {
#         "input": 2.50,
#         "output": 10.00,
#     },
#     "gpt-4o-mini": {
#         "input": 0.15,
#         "output": 0.60,
#     },
#     "claude-3.5-sonnet": {
#         "input": 3.00,
#         "output": 15.00,
#     },
#     "claude-3-haiku": {
#         "input": 0.25,
#         "output": 1.25,
#     },
#     "llama-3.1-70B": {
#         "input": 0.88,
#         "output": 0.88,
#     },
#     "llama-3.1-405B": {
#         "input": 3.50,
#         "output": 3.50,
#     },
# }
# for model, price in llm_pricing.items():
#     # this is an overestimate b/c I should use nowhere near this number of output tokens
#     input_cost = price["input"] * total_tokens / 1_000_000
#     output_cost = price["output"] * total_tokens / 1_000_000
#     cost = input_cost + output_cost
#     print(f"  {model} will cost ${cost:,.4f}")

# %%
