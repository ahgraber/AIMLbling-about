# %%
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import typing as t

from dotenv import load_dotenv
from IPython.display import display
from pydantic import BaseModel
import tiktoken
from tqdm.auto import tqdm

import pandas as pd

import torch

# NOTE: RAGAS uses langchain as primary integration, so use it for convenience
from langchain.text_splitter import TokenTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document as LCDoc
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
import openai
from ragas import RunConfig
from ragas.cost import CostCallbackHandler, TokenUsage, get_token_usage_for_anthropic, get_token_usage_for_openai
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, NodeType
from ragas.testset.synthesizers import (
    AbstractQuerySynthesizer,
    ComparativeAbstractQuerySynthesizer,
    SpecificQuerySynthesizer,
)

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
repo = subprocess.check_output(  # NOQA: S603
    ["git", "rev-parse", "--show-toplevel"],  # NOQA: S607
    cwd=Path(__file__).parent,
    encoding="utf-8",
).strip()
repo = Path(repo).resolve()

datadir = Path(__file__).parent / "data"


# %%
def check_torch_device():
    """Check which device pytorch will use."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Found pytorch device '{device.type}'")
    return device


device = check_torch_device()

# %%
_ = load_dotenv()

# # Local uses LMStudio to host a local OpenAI-compatible service
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")  # "http://localhost:1234/v1"

# OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

LLM_KWARGS = {
    "temperature": 0,
    "max_tokens": 2048,  # max output tokens
}
RESILIENCE_KWARGS = {
    "max_retries": 10,
    "timeout": 60,
}

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
        # "text": "search_document: ",  # use key "text" to embed documents
        # "query": "search_query: ",  # use key "query" to embed queries
        "clustering": "clustering: ",
        # "classification": "classification: ",
    },
    # "default_prompt_name": "",
}
cluster_em = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
    model_kwargs={
        **nomic_embedding_kwargs,
        **{"default_prompt_name": "clustering"},
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt_name": "clustering",
    },
)
# document_em = HuggingFaceEmbeddings(
#     model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
#     model_kwargs={
#         **nomic_embedding_kwargs,
#         **{"default_prompt_name": "text"},
#     },
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "prompt_name": "text",
#     },
# )
# query_em = HuggingFaceEmbeddings(
#     model_name="nomic-ai/nomic-embed-text-v1.5",  # default context 2048 tokens
#     model_kwargs={
#         **nomic_embedding_kwargs,
#         **{"default_prompt_name": "query"},
#     },
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "prompt_name": "query",
#     },
# )


# %%
def llama_finished_parser(response: LLMResult) -> bool:
    """Check TogetherAI/Llama response for successful generation."""
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]
        if resp.generation_info is not None:
            # generation_info is provided - so we parse that

            # OpenAI uses "finish_reason": "stop"
            # Together/Llama uses "finish_reason": "stop" or "eos"
            if resp.generation_info.get("finish_reason") is not None:
                is_finished_list.append(resp.generation_info.get("finish_reason") in ["stop", "eos"])
            # provide more conditions here: https://github.com/explodinggradients/ragas/issues/1548

        # if generation_info is empty, we parse the response_metadata
        # this is less reliable
        elif t.cast(ChatGeneration, resp).message is not None:
            resp_message: BaseMessage = t.cast(ChatGeneration, resp).message
            if resp_message.response_metadata.get("finish_reason") is not None:
                is_finished_list.append(resp_message.response_metadata.get("finish_reason") in ["stop", "eos"])
            elif resp_message.response_metadata.get("stop_reason") is not None:
                is_finished_list.append(resp_message.response_metadata.get("stop_reason") == "end_turn")
        # default to True
        else:
            is_finished_list.append(True)
    return all(is_finished_list)


# %%
providers = {
    "local": {
        "llm": ChatOpenAI(
            base_url=LOCAL_BASE_URL,
            # base_url="http://localhost:1234/v1",
            api_key=LOCAL_API_KEY,
            # model="phi-3.1-mini-128k-instruct",
            model="mistral-nemo-instruct-2407",
            **LLM_KWARGS,
            max_retries=10,
            timeout=120,
        ),
        # "em": OpenAIEmbeddings(
        #     base_url=LOCAL_BASE_URL,
        #     # base_url="http://localhost:1234/v1",
        #     api_key=LOCAL_API_KEY,
        #     model="nomic-embed-text-v1.5",
        #     check_embedding_ctx_length=False,  # ref: https://github.com/langchain-ai/langchain/issues/21318
        #     **RESILIENCE_KWARGS,
        # ),
        "em": cluster_em,
    },
    "openai": {
        "llm": ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
            # deployment="text-embedding-3-small",
            **RESILIENCE_KWARGS,
        ),
    },
    "anthropic": {
        "llm": ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            **LLM_KWARGS,
            **RESILIENCE_KWARGS,
        ),
        "em": VoyageAIEmbeddings(
            voyage_api_key=VOYAGE_API_KEY,
            model="voyage-3-lite",
            batch_size=7,  # ref: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/voyageai/
            # **resilience_kwargs,
        ),
    },
    "together": {
        "llm": ChatOpenAI(  # Langchain-Together is based on Langchain-OpenAI; might as well use source
            base_url="https://api.together.xyz/v1",
            api_key=TOGETHER_API_KEY,
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
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            api_key=TOGETHER_API_KEY,
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
    kg = KnowledgeGraph().load(datadir / "ragas_knowledgegraph.json")
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

    dataset.to_jsonl(datadir / f"ragas_dataset_{provider}.jsonl")

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
