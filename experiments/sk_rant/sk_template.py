# ruff: NOQA: F704 # allow naked 'await' since using as ipynb

# %%
import asyncio
import copy
import json
import logging
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import InputVariable, KernelPromptTemplate, PromptTemplateConfig

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
async def render_prompt_from_template(
    kernel: Kernel,
    plugin_name: str,
    function_name: str,
    arguments: KernelArguments,
):
    """Render the full prompt from the template and kernel arguments.

    KernelArguments contain the kwargs for the template variables (among other info).
    """
    func = kernel.get_function_from_fully_qualified_function_name(f"{plugin_name}-{function_name}")

    rendered = await func.prompt_template.render(kernel, arguments=arguments)
    return rendered


async def render_api_call(
    kernel: Kernel,
    service_id: str,
    plugin_name: str,
    function_name: str,
    arguments: KernelArguments,
    streaming: bool = False,
    simple: bool = True,
):
    """
    Given template function, return (expected) API call.

    API call is rendered from complete ChatHistory.
    ChatHistory is derived from fully-rendered prompt.

    References
    ----------
    - [KernelFunctionFromPrompt._invoke_internal()](https://github.com/microsoft/semantic-kernel/blob/77aa4e32829dfe55d8924b91b9a1eb567147ec8c/python/semantic_kernel/functions/kernel_function_from_prompt.py#L161)
    - [OpenAIChatCompletionBase.get_chat_message_contents()](https://github.com/microsoft/semantic-kernel/blob/9363a190d4f1f1e674167370cf4d67211c796e33/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_chat_completion_base.py#L76)
    """
    # Create a copy of the settings to avoid modifying the originals
    settings = copy.deepcopy(arguments.execution_settings[service_id])

    # render prompt from template
    rendered = await render_prompt_from_template(
        kernel,
        plugin_name=plugin_name,
        function_name=function_name,
        arguments=arguments,
    )

    # create (internal) history from rendered prompt
    _history = ChatHistory().from_rendered_prompt(rendered)

    # generate "settings", aka the API call args
    service = kernel.get_service(service_id=service_id)
    service._prepare_settings(
        settings=settings,
        chat_history=_history,
        stream_request=streaming,
        kernel=kernel,
    )

    if simple:
        settings.extension_data = {}

    return settings.dict()


# %%
# load .env file specific to desired environment/resources (see .env.sample)
_ = load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # need a working key or to mock the endpoint

SERVICE_ID = "gpt-4o"

# %%
kernel = Kernel()

chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    service_id=SERVICE_ID,  # arbitrary for developer reference
)
kernel.add_service(chat_service)

# %%
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=SERVICE_ID).dict()
req_settings.update(  # inplace update
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
req_settings = OpenAIChatPromptExecutionSettings(**req_settings)


# %%
system_message = "You are a helpful AI assistant."
chat_history = ChatHistory()
chat_history.add_user_message("Hi, who are you?")
chat_history.add_assistant_message("I am a helpful AI assistant.")

user_request = "How will AI influence the Project Management role?"

# %% [markdown]
# ## PromptTemplateConfig

# %%
# template = """

# Previous information from chat:
# {{$chat_history}}

# User: {{$user_request}}
# Assistant:
# """

template = """
User: {{$user_request}}
Assistant:
"""


prompt_template_config = PromptTemplateConfig(
    template=template,
    name="template",
    description="Chat with the assistant",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(
            name="system_message",
            description="The system message",
            is_required=False,
            default="",
        ),
        InputVariable(
            name="chat_history",
            description="The conversation history",
            is_required=False,
            default="",
        ),
        InputVariable(
            name="user_request",
            description="The user's request",
            is_required=True,
        ),
    ],
    execution_settings=req_settings,
)
prompt_template = KernelPromptTemplate(
    prompt_template_config=prompt_template_config,
)

# add chat function to kernel
template_function = kernel.add_function(
    plugin_name="demo",
    function_name="template",
    prompt_template_config=prompt_template_config,
)

rendered = await prompt_template.render(
    kernel,
    arguments=KernelArguments(
        req_settings,
        chat_history=chat_history,
        user_request=user_request,
    ),
)
print(rendered)


# %% [markdown]
# ChatHistory provides a record of all messages sent/received in the conversation.
# It is analogous to the messages array in the OpenAI API.
#
# ChatHistory.messages is an array of ChatMessageContent objects
# ChatMessageContent retains role, text, and other metadata (if they exist)
#
# ChatHistory is rendered internally with XML, but can be serialized to JSON for OpenAI API compatibility.
#
# Since prompt template and history are rendered as XML, we can convert between prompt-template-with-history and messages-array style format

# %%
# create history from rendered prompt
history = ChatHistory.from_rendered_prompt(rendered)
# Since no XML is present in the template, everything is crammed into a `user` message

# %%
# view history as messages array
# print(json.dumps(chat_service._prepare_chat_history_for_request(history), indent=2))
print(json.dumps([message.to_dict() for message in history.messages], indent=2))

# %%
# view history as complete/full messages array
print(history.serialize())


# %%
# convert history back to prompt
print(history.to_prompt())

# %% [markdown]
# ## Template to API

# %%
await render_api_call(
    kernel=kernel,
    service_id=SERVICE_ID,
    plugin_name="demo",
    function_name="template",
    arguments=KernelArguments(
        req_settings,
        chat_history=chat_history,
        user_request=user_request,
    ),
    streaming=False,
    simple=True,
)


# %% [markdown]
# ## Template Prompt

# %%
rendered = await render_prompt_from_template(
    kernel=kernel,
    plugin_name="demo",
    function_name="template",
    arguments=KernelArguments(
        req_settings,
        system_message=system_message,
        chat_history=chat_history,
        user_request=user_request,
    ),
)
print(rendered)

# %%
call_args = await render_api_call(
    kernel=kernel,
    service_id=SERVICE_ID,
    plugin_name="demo",
    function_name="template",
    arguments=KernelArguments(
        req_settings,
        system_message=system_message,
        chat_history=chat_history,
        user_request=user_request,
    ),
    streaming=False,
    simple=True,
)
print(json.dumps(call_args, indent=2))

# %%
