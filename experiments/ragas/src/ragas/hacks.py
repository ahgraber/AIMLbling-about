import itertools
import logging
import typing as t

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

logger = logging.getLogger(__name__)


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
