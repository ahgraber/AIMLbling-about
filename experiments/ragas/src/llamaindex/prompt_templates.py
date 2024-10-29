import logging
import textwrap

from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

logger = logging.getLogger(__name__)

# define baseline no-RAG prompt similar to the one LlamaIndex uses
BASELINE_QA_PROMPT_TMPL = textwrap.dedent("""
    Given prior knowledge, answer the query.
    Query: {query_str}
    Answer: """).lstrip()
BASELINE_QA_PROMPT = PromptTemplate(
    BASELINE_QA_PROMPT_TMPL,
    prompt_type=PromptType.QUESTION_ANSWER,
)

# this is the llamaindex prompt for default RAG
DEFAULT_TEXT_QA_PROMPT_TMPL = textwrap.dedent("""
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: """).lstrip()
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    prompt_type=PromptType.QUESTION_ANSWER,
)
