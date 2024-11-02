# ruff: NOQA: F704 # allow naked 'await' since using as ipynb

# %%
import asyncio
from collections import deque
import copy
import datetime
import json
import logging
import os
import re
import sys
import textwrap
import types

from dotenv import load_dotenv
import networkx as nx

import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import ContentFilterAIException
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
    OpenAIPromptExecutionSettings,
    OpenAITextPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_model_types import OpenAIModelTypes
from semantic_kernel.contents import ChatMessageContent, ImageContent, TextContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions import ServiceResponseException
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel_pydantic import KernelBaseModel

import matplotlib.pyplot as plt

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
# load .env file specific to desired environment/resources (see .env.sample)
_ = load_dotenv(".env")

OPENAI_API_KEY = os.getenv("_OPENAI_API_KEY")  # need a working key or to mock the endpoint

SERVICE_ID = "chat"

# %%
kernel = sk.Kernel()

# Add the Azure OpenAI chat completion service
chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    service_id=SERVICE_ID,  # arbitrary for developer reference
)
kernel.add_service(chat_service)

# get standard settings for the service
_ec = kernel.get_prompt_execution_settings_from_service_id(service_id=SERVICE_ID).dict()
# inplace update for custom settings
_ec.update(
    {
        "max_tokens": None,
        "temperature": 0,
        # 'top_p': None,
        # 'frequency_penalty': None,
        # 'presence_penalty': None,
        # 'seed': None,
        # 'stream': False,
        # 'logit_bias': None,
        # 'number_of_responses': None,
        # 'stop': None,
        # 'user': None,
        # 'response_format': None,
        # 'function_call': None,
        # "function_choice_behavior":FunctionChoiceBehavior.Auto(filters={"excluded_plugins": []}),
        # 'functions': None,
        # 'messages': None,
        # 'tools': None,
        # 'tool_choice': None,
    }
)
execution_config = OpenAIChatPromptExecutionSettings(**_ec)
options = execution_config.prepare_settings_dict()

# %%
system_msg = "You are a helpful AI assistant."

# create chat plugin/function that uses chat history
history_chat = kernel.add_function(
    plugin_name="ChatBot",
    function_name="chat",
    prompt="{{$history}}",
    prompt_execution_settings=execution_config,
)

history = ChatHistory()
history.add_system_message(system_msg)

user_input = "How will AI effect projects in ways that are meaningful to the job that project managers do?"
history.add_user_message(user_input)


# %%
def wrap_or_break(text: str, width: int = 20, split_on: str = " "):
    """Attempt to wrap text at width or nearest `split_on` character.

    Returns a list of output lines, without final newlines.
    """
    d = deque(text.split("."))
    x = []
    while d:
        a = d.popleft()
        try:
            b = d.popleft()
        except IndexError:
            x.append(a)
            break

        c = f"{a}.{b}"
        if len(c) <= width:
            x.append(c)
        else:
            x.append(a)
            d.appendleft(b)

    return x


class CallGraphTracer:
    """Trace all calls involved in function execution."""

    def __init__(self, module_filter: str | None = None):
        self.module_filter = module_filter
        self.call_graph = nx.MultiDiGraph()
        self.current_func = None

    def __enter__(self):  # NOQA: D105
        def trace(frame, event, arg):
            if event == "call":
                module_name = frame.f_globals.get("__name__")
                func_name = frame.f_code.co_name

                caller = self.current_func
                callee = f"{module_name}.{func_name}"
                if caller and (
                    not self.module_filter  # no filter, take all
                    or (
                        caller.startswith(self.module_filter)  # if filter, only use module prefix
                        or callee.startswith(self.module_filter)
                    )
                ):
                    self.call_graph.add_edge(caller, callee)
                self.current_func = callee
            elif event == "return":
                self.current_func = None
            return trace

        sys.settrace(trace)
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # NOQA: D105
        sys.settrace(None)

    def draw_call_graph(self, figsize: tuple[int, int], bfs: bool = True, start: str | None = None):
        """Visualize call graph."""
        call_graph = copy.deepcopy(self.call_graph)

        if bfs:
            if start is None:
                start = next(iter(call_graph.nodes))

            call_graph = nx.bfs_tree(call_graph, source=start, reverse=False)
            pos = nx.bfs_layout(call_graph, start=start, align="vertical")

        else:
            pos = nx.spring_layout(call_graph)

        plt.figure(figsize=figsize)
        nx.draw(
            call_graph,
            pos,
            arrowsize=5,
            node_shape="s",
            node_color="lightsteelblue",
            node_size=750,
            margins=0.05,
            # with_labels=True,
        )
        node_labels = {n: "\n.".join(wrap_or_break(n)) for n, _ in call_graph.nodes.items()}
        nx.draw_networkx_labels(
            call_graph,
            pos,
            node_labels,
            font_size=10,
            font_weight="normal",
        )

        plt.title("Function Call Graph")
        plt.axis("off")
        # plt.tight_layout() # "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect."
        plt.show()


# %%
with CallGraphTracer(module_filter="semantic_kernel") as tracer:
    result = await kernel.invoke(  # NOQA: F704
        plugin_name="ChatBot",
        function_name="chat",
        arguments=KernelArguments(
            settings=execution_config,
            history=history,
        ),
    )


# %%
tracer.draw_call_graph(figsize=(18, 6), bfs=True)

# %%
tracer.draw_call_graph(figsize=(12, 12), bfs=False)
