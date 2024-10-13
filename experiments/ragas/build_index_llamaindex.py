# %%
from itertools import chain
import json
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from IPython.display import display

# RAGAS uses langchain for its InMemoryDocStore used for TestsetGenerator
from langchain.text_splitter import TokenTextSplitter

# use Llamaindex for the rest of the integrations
from llama_index.core import Document as LlamaDoc, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.docstore import Document as RagasDoc, InMemoryDocumentStore, Node as RagasNode
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.generator import TestsetGenerator
import tiktoken

import pandas as pd

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
# def check_torch_device():
#     """Check which device pytorch will use."""
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps:0")
#     else:
#         device = torch.device("cpu")

#     logger.info(f"Found pytorch device '{device.type}'")
#     return device


# device = check_torch_device()

# %%
_ = load_dotenv()
# HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# %% [markdown]
# ## Load files
#
# Source corpus will be blog posts and markdown files from my homelab/gitops repo


# %%
def hugo_title_to_h1(text: str):
    """Extract title hugo front matter section and convert to markdown H1."""
    frontmatter = re.match(r"(?P<frontmatter>---[\s\S]*?---)", text)

    try:
        frontmatter = frontmatter["frontmatter"]
    except TypeError:
        # no frontmatter; return without changes
        return text

    title = re.match(r"[\s\S]*title: (?P<title>.*)\n", frontmatter)
    try:
        title = f"# {title['title']}\n\n"  # ensure title has trailing newlines
    except TypeError:
        logger.info("Could not parse title from frontmatter")
        logger.debug({frontmatter["frontmatter"]})
        title = ""
    else:
        text = text.replace(frontmatter, title)
        text = re.sub(r"\n{3,}", "\n\n", text)  # clean up overzealous newlines
        return text


blog_files = list((repo / "content" / "blog").rglob("*.md"))
blog_docs = [
    LlamaDoc(
        text=hugo_title_to_h1(f.read_text()),
        metadata={
            "name": f.parent.name,
            "filename": str(f),
            "category": "blog",
        },
    )
    for f in blog_files
]

# %%
k3s_files = list(
    chain(
        (Path.home() / "_code" / "homelab-gitops-k3s").glob("readme.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "docs").rglob("*.md"),
        (Path.home() / "_code" / "homelab-gitops-k3s" / "kubernetes").rglob("*.md"),
    )
)
k3s_docs = [
    LlamaDoc(
        text=f.read_text(),
        metadata={
            "name": f"{f.parent.name}_{f.name}" if "readme" in f.name.lower() else f.name,
            "filename": str(f),
            "category": "k3s",
        },
    )
    for f in k3s_files
]

# %%
corpus = blog_files + k3s_files
documents = blog_docs + k3s_docs
assert len(corpus) == len(documents), "Error: file count in corpus does not match document count"  # NOQA: S101
print(f"{len(documents)=}")

# %%
tokenizer = tiktoken.encoding_for_model("gpt-4")
total_tokens = sum(len(tokenizer.encode(c.read_text())) for c in corpus)
print(f"Corpus contains {total_tokens} tokens")

# %%
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(documents)
print(f"{len(nodes)=}")


# %%
tokenizer = tiktoken.encoding_for_model("gpt-4")
token_counts = [len(tokenizer.encode(node.text)) for node in nodes]
pd.Series(token_counts).plot(kind="hist")

MAX_EMBEDDING_LENGTH = 8190
longest_node = max(token_counts)
print(f"Longest node is {longest_node} tokens")
assert (  # NOQA: S101
    longest_node < MAX_EMBEDDING_LENGTH
), f"Error: some chunks are longer than the max embedding length {MAX_EMBEDDING_LENGTH}"

# %%
print("Embedding:")
emb_pricing = {
    "text-embedding-3-small": {
        "input": 0.02,
    },
    "text-embedding-3-large": {
        "input": 0.13,
    },
    "voyage-3": {  # anthropic partner
        "input": 0.06,
    },
    "voyage-3-lite": {  # anthropic partner
        "input": 0.02,
    },
    "M2-BERT-80M-8K-Retrieval": {  # together.ai
        "input": 0.008,
    },
}
for model, price in emb_pricing.items():
    input_cost = price["input"] * total_tokens / 1_000_000
    cost = input_cost
    print(f"  {model} will cost ${cost:,.4f}")

# Ref: https://huggingface.co/spaces/philschmid/llm-pricing
# PPM -> price-per-million
print("Generation:")
llm_pricing = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "claude-3.5-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "llama-3.1-70B": {
        "input": 0.90,
        "output": 0.90,
    },
    "llama-3.1-405B": {
        "input": 3.50,
        "output": 3.50,
    },
}
for model, price in llm_pricing.items():
    input_cost = price["input"] * total_tokens / 1_000_000
    output_cost = price["output"] * total_tokens / 1_000_000  # this is an overestimate of output token use
    cost = input_cost + output_cost
    print(f"  {model} will cost ${cost:,.4f}")


# %% [markdown]
# ## Set up models with LlamaIndex adapters for easy integration w/ RAGAS

# %%
llm_kwargs = {
    "temperature": 0,
    "max_tokens": 4096,  # 1024,
}
resilience_kwargs = {
    "max_retries": 10,
    "timeout": 60,
}
providers = {
    "openai": {
        "llm": OpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            **llm_kwargs,
            **resilience_kwargs,
        ),
        "em": OpenAIEmbedding(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
            **resilience_kwargs,
        ),
    },
    "anthropic": {
        "llm": Anthropic(
            api_key=ANTHROPIC_API_KEY,
            model="claude-3-5-sonnet-20240620",
            **llm_kwargs,
            **resilience_kwargs,
        ),
        "em": VoyageEmbedding(
            voyage_api_key=VOYAGE_API_KEY,
            model_name="voyage-3-lite",
            # **resilience_kwargs,
        ),
    },
    "together": {
        "llm": TogetherLLM(
            api_key=TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            **llm_kwargs,
            **resilience_kwargs,
        ),
        "em": TogetherEmbedding(
            api_key=TOGETHER_API_KEY,
            model_name="togethercomputer/m2-bert-80M-8k-retrieval",
            **resilience_kwargs,
        ),
    },
}


# %% [markdown]
# ## Generate Benchmark Dataset
#
# Dataset should be independent of any postprocessing (i.e., chunking, vectorizing, reranking, prompt instruction tuning, etc).
# Theoretically, the dataset should also be model-independent, but that's why we're running this experiment.
#
# This means the dataset is preferably generated at the _document_ level, without the influence of the processing pipeline
#
# RAGAS does not make this possible and enforces chunking during the test set generation

# %%
# for provider in providers:
provider = "anthropic"
llm = providers[provider]["llm"]
em = providers[provider]["em"]

# %%
# RunConfig defines defaults
# fmt: off
run_config = RunConfig(
    timeout=60,     # default 60
    max_retries=10, # default 10
    max_wait=60,    # default 60
    max_workers=4,  # default 16
)
# fmt: on
distributions = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}

# %%
# By default, RAGAS will create a docstore that will chunk and embed documents.
# Default docstore uses short tokenizer chunks; but our embedding models allow longer
docstore = InMemoryDocumentStore(
    splitter=TokenTextSplitter(chunk_size=MAX_EMBEDDING_LENGTH, chunk_overlap=0),
    embeddings=LlamaIndexEmbeddingsWrapper(em),
    extractor=KeyphraseExtractor(LlamaIndexLLMWrapper(llm)),
    run_config=run_config,
)

# %%
# Manually fill docstore with existing nodes from llamaindex

# # docstore does not actually retain full documents
# ragas_docs = [r=RagasDoc.from_llamaindex_document(d) for d in documents]

# NOTE: this will run embeddings and keyphrase extraction for all nodes!
docstore.add_nodes([RagasNode.from_llamaindex_document(node) for node in nodes])

# %%
generator = TestsetGenerator.from_llama_index(
    generator_llm=llm,
    critic_llm=llm,
    embeddings=em,
    docstore=docstore,
    run_config=run_config,
)


# save node/embeddings from docstore
with (datadir / f"docstore_{provider}.jsonl").open("w") as f:
    for id_, node in generator.docstore.node_map.items():
        nodemap = {
            "node_id": id_,
            # **node.to_json()
            "filename": node.filename,
            "document_id": node.doc_id,
            "metadata": node.metadata,
            "text": node.page_content,
            "embedding": node.embedding,
            "keyphrases": node.keyphrases,
            "relationships": {
                "prev": node.relationships["prev"].doc_id if node.relationships.get("prev", None) else None,
                "next": node.relationships["next"].doc_id if node.relationships.get("next", None) else None,
            },
        }
        json.dump(nodemap, f)
        f.write("\n")

# # save generator object
# with (datadir / f"generator_{provider}.pkl").open("wb") as f:
#     pickle.dump(generator, f)
#     # removing unpickleable private attribute _client, _aclient

# %%
# generate testset

# call `generator.generate` directly
# don't do `generator.generate_with_...` because we've custom-filled the docstore above
# NOTE: this will call the generator and critic LLM(s) to generate the Q/A pairs
testset = generator.generate(
    test_size=100,
    distributions=distributions,
)

# %%
test_df = testset.to_pandas()
test_df["provider"] = provider
try:
    test_df["model"] = llm.model
except AttributeError:
    test_df["model"] = llm.model_name
try:
    test_df["embed"] = em.model
except AttributeError:
    test_df["embed"] = em.model_name

display(test_df)

# %%
# save testset
test_df.to_csv(datadir / f"ragas_test_set_{provider}.csv")

# save prompts
generator.save(
    evolutions=list(distributions.keys()),
    cache_dir=datadir / "prompts",
)

# %%
