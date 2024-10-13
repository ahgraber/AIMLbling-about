# %%
from itertools import chain
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from IPython.display import display

# NOTE: RAGAS uses langchain as primary integration, so use it for convenience
from langchain.text_splitter import TokenTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document as lcDocument
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.docstore import Document as rDocument, InMemoryDocumentStore
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
    lcDocument(
        page_content=hugo_title_to_h1(f.read_text()),
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
    lcDocument(
        page_content=f.read_text(),
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
print(f"{len(documents)=}")

# %%
tokenizer = tiktoken.encoding_for_model("gpt-4")

total_tokens = sum(len(tokenizer.encode(c.read_text())) for c in corpus)

print(f"Corpus contains {total_tokens} tokens")

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
    # this is an overestimate b/c I should use nowhere near this number of output tokens
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
    # this is an overestimate b/c I should use nowhere near this number of output tokens
    input_cost = price["input"] * total_tokens / 1_000_000
    output_cost = price["output"] * total_tokens / 1_000_000
    cost = input_cost + output_cost
    print(f"  {model} will cost ${cost:,.4f}")


# %%
class CustomMarkdownSyntaxTextSplitter:
    """Split markdown with ExperimentalMarkdownSyntaxTextSplitter.

    Approximates a LangChain TextSplitter object with split_documents and transform_documents methods.
    """

    def __init__(
        self,
        headers_to_split_on: Union[List[Tuple[str, str]], None] = None,
        return_each_line: bool = False,
        strip_headers: bool = True,
    ) -> None:
        self.headers_to_split_on = headers_to_split_on
        self.return_each_line = return_each_line
        self.strip_headers = strip_headers

    def split_documents(self, documents: Iterable[lcDocument]) -> List[lcDocument]:
        """Split documents."""
        # as of v0.2.16, ExperimentalMarkdownSyntaxTextSplitter.split_text() returns a list of Documents _and retains all documents as self.chunks_
        # This means we need a new instance per document we are splitting

        subdocuments = []
        for doc in documents:
            emsts = ExperimentalMarkdownSyntaxTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                return_each_line=self.return_each_line,
                strip_headers=self.strip_headers,
            )

            # self.split_texts -> List[Document]
            chunks = emsts.split_text(doc.page_content)
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)

            subdocuments.extend(chunks)

        return subdocuments

    def transform_documents(self, documents: Sequence[lcDocument]) -> List[lcDocument]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))


# %%
cmsts = CustomMarkdownSyntaxTextSplitter()
nodes = cmsts.transform_documents(documents)
print(f"{len(nodes)=}")

# %%
tokenizer = tiktoken.encoding_for_model("gpt-4")

token_counts = [len(tokenizer.encode(node.page_content)) for node in nodes]
pd.Series(token_counts).plot(kind="hist")

MAX_EMBEDDING_LENGTH = 8190
longest_node = max(token_counts)
assert (  # NOQA: S101
    longest_node < MAX_EMBEDDING_LENGTH
), f"Error: some chunks are longer than the max embedding length {MAX_EMBEDDING_LENGTH}"


# %% [markdown]
# ## Set up models with Langchain adapters for easy integration w/ RAGAS

# %%
providers = {
    "openai": {
        "llm": ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=7,
            # other params...
        ),
        "em": OpenAIEmbeddings(
            model="text-embedding-3-small",
            deployment="text-embedding-3-small",
            # dimensions=...
        ),
    },
    "anthropic": {
        "llm": ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=7,
            # other params...
        ),
        "em": VoyageAIEmbeddings(model="voyage-3-lite"),
    },
    "together": {
        # "llm": ChatTogether(
        #     model="meta-llama/Llama-3-70b-chat-hf",
        #     temperature=0,
        #     max_tokens=1024,
        #     timeout=None,
        #     max_retries=7,
        #     # other params...
        #     api_key=TOGETHER_API_KEY,
        # ),
        "llm": ChatOpenAI(  # Langchain-Together is based on Langchain-OpenAI; might as well use source
            base_url="https://api.together.xyz/v1",
            api_key=TOGETHER_API_KEY,
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=10,  # low tier == rate limits
        ),
        "em": TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            api_key=TOGETHER_API_KEY,
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
provider = "openai"
llm = providers[provider]["llm"]
em = providers[provider]["em"]

# %%
# RAGAS will create a docstore for vector lookups if we do not provide one;
# Default docstore uses short tokenizer chunks
# Prefer to use custom chunking vs the default
docstore = InMemoryDocumentStore(
    splitter=TokenTextSplitter(chunk_size=MAX_EMBEDDING_LENGTH, chunk_overlap=0),
    embeddings=LangchainEmbeddingsWrapper(em),
    extractor=KeyphraseExtractor(LangchainLLMWrapper(llm)),
    run_config=RunConfig(),
)

# docstore = InMemoryDocumentStore(
#     splitter=CustomMarkdownSyntaxTextSplitter(),
#     embeddings=LangchainEmbeddingsWrapper(em),
#     extractor=KeyphraseExtractor(LangchainLLMWrapper(llm)),
#     run_config=RunConfig(),
# )

generator = TestsetGenerator.from_langchain(
    generator_llm=llm,
    critic_llm=llm,
    embeddings=em,
    docstore=docstore,
)

# %%
# generate testset
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=100,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)

# %%
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
test_df.to_csv(datadir / f"ragas_test_set_{provider}.csv")

# %%
