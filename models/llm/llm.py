import logging
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    AssistantPromptMessage,
)

import requests
import json

from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class YuanjingLargeLanguageModel(OAICompatLargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._add_custom_parameters(credentials)

        # print(f"model: {model}, credentials: {credentials}, prompt_messages: {prompt_messages}, model_parameters: {model_parameters}, tools: {tools}, stop: {stop}, stream: {stream}, user: {user} -----")

        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)


    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke llm completion model

        :param model: model name
        :param credentials: credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8",
        }
        extra_headers = credentials.get("extra_headers")
        if extra_headers is not None:
            headers = {
                **headers,
                **extra_headers,
            }

        api_key = credentials.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = credentials["endpoint_url"]
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"

        response_format = model_parameters.get("response_format")
        if response_format:
            if response_format == "json_schema":
                json_schema = model_parameters.get("json_schema")
                if not json_schema:
                    raise ValueError("Must define JSON Schema when the response format is json_schema")
                try:
                    schema = json.loads(json_schema)
                except Exception:
                    raise ValueError(f"not correct json_schema format: {json_schema}")
                model_parameters.pop("json_schema")
                model_parameters["response_format"] = {"type": "json_schema", "json_schema": schema}
            else:
                model_parameters["response_format"] = {"type": response_format}
        elif "json_schema" in model_parameters:
            del model_parameters["json_schema"]

        data = {"model": model, "stream": stream, **model_parameters}

        completion_type = LLMMode.value_of(credentials["mode"])

        if completion_type is LLMMode.CHAT:
            endpoint_url = urljoin(endpoint_url, "chat/completions")
            data["messages"] = [self._convert_prompt_message_to_dict(m, credentials) for m in prompt_messages]
        elif completion_type is LLMMode.COMPLETION:
            endpoint_url = urljoin(endpoint_url, "completions")
            data["prompt"] = prompt_messages[0].content
        else:
            raise ValueError("Unsupported completion type for model configuration.")

        # annotate tools with names, descriptions, etc.
        function_calling_type = credentials.get("function_calling_type", "no_call")
        formatted_tools = []
        if tools:
            if function_calling_type == "function_call":
                data["functions"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            elif function_calling_type == "tool_call":
                data["tool_choice"] = "auto"

                for tool in tools:
                    formatted_tools.append(PromptMessageFunction(function=tool).model_dump())

                data["tools"] = formatted_tools

        if stop:
            data["stop"] = stop

        if user:
            data["user"] = user

        response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300), stream=stream)

        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = "utf-8"

        if response.status_code != 200:
            raise InvokeError(f"API request failed with status code {response.status_code}: {response.text}")

        if stream:
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)

        return self._handle_generate_response(model, credentials, response, prompt_messages)


    def _handle_generate_stream_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> Generator:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: model credentials
        :param response: streamed response
        :param prompt_messages: prompt messages
        :return: llm response chunk generator
        """
        full_assistant_content = ""
        chunk_index = 0

        def create_final_llm_result_chunk(
            id: Optional[str],
            index: int,
            message: AssistantPromptMessage,
            finish_reason: str,
            usage: dict,
        ) -> LLMResultChunk:
            # calculate num tokens
            prompt_tokens = usage.get("prompt_tokens") if usage else 0
            if prompt_tokens is None:
                assert prompt_messages[0].content is not None
                prompt_tokens = self._num_tokens_from_string(model, prompt_messages[0].content)
            completion_tokens = usage.get("completion_tokens") if usage else 0
            if completion_tokens is None:
                completion_tokens = self._num_tokens_from_string(model, full_assistant_content)

            # transform usage
            usage_obj = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

            return LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=index,
                    message=message,
                    finish_reason=finish_reason,
                    usage=usage_obj,
                ),
            )

        # delimiter for stream response, need unicode_escape
        import codecs

        delimiter = credentials.get("stream_mode_delimiter", "\n\n")
        delimiter = codecs.decode(delimiter, "unicode_escape")

        tools_calls: list[AssistantPromptMessage.ToolCall] = []

        def increase_tool_call(new_tool_calls: list[AssistantPromptMessage.ToolCall]):
            def get_tool_call(tool_call_id: str):
                if not tool_call_id:
                    return tools_calls[-1]

                tool_call = next(
                    (tool_call for tool_call in tools_calls if tool_call.id == tool_call_id),
                    None,
                )
                if tool_call is None:
                    tool_call = AssistantPromptMessage.ToolCall(
                        id=tool_call_id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="", arguments=""),
                    )
                    tools_calls.append(tool_call)

                return tool_call

            for new_tool_call in new_tool_calls:
                # get tool call
                tool_call = get_tool_call(new_tool_call.function.name)
                # update tool call
                if new_tool_call.id:
                    tool_call.id = new_tool_call.id
                if new_tool_call.type:
                    tool_call.type = new_tool_call.type
                if new_tool_call.function.name:
                    tool_call.function.name = new_tool_call.function.name
                if new_tool_call.function.arguments:
                    tool_call.function.arguments += new_tool_call.function.arguments

        finish_reason = None  # The default value of finish_reason is None
        message_id, usage = None, None
        for chunk in response.iter_lines(decode_unicode=True, delimiter=delimiter):
            chunk = chunk.strip()
            print(f"origin chunk: {chunk} -------")
            if chunk:
                # ignore sse comments
                if chunk.startswith(":"):
                    continue
                # 关键，实现上不同，没有空格
                decoded_chunk = chunk.strip().removeprefix("data:").lstrip()
                if decoded_chunk == "[DONE]":  # Some provider returns "data: [DONE]"
                    continue

                try:
                    chunk_json: dict = json.loads(decoded_chunk)

                    print(f"chunk_json: {chunk_json} --- @@@@")
                # stream ended
                except json.JSONDecodeError:
                    print(f"chunk_json_parse_error: --- !!!")
                    yield create_final_llm_result_chunk(
                        id=message_id,
                        index=chunk_index + 1,
                        message=AssistantPromptMessage(content=""),
                        finish_reason="Non-JSON encountered.",
                        usage=usage or {},
                    )
                    break
                if chunk_json and (u := chunk_json.get("usage")):
                    usage = u
                if not chunk_json or len(chunk_json["choices"]) == 0:
                    continue

                choice = chunk_json["choices"][0]
                finish_reason = chunk_json["choices"][0].get("finish_reason")
                message_id = chunk_json.get("id")
                chunk_index += 1

                if "delta" in choice:
                    delta = choice["delta"]
                    delta_content = delta.get("content")

                    assistant_message_tool_calls = None

                    if "tool_calls" in delta and credentials.get("function_calling_type", "no_call") == "tool_call":
                        assistant_message_tool_calls = delta.get("tool_calls", None)
                    elif (
                        "function_call" in delta
                        and credentials.get("function_calling_type", "no_call") == "function_call"
                    ):
                        assistant_message_tool_calls = [
                            {
                                "id": "tool_call_id",
                                "type": "function",
                                "function": delta.get("function_call", {}),
                            }
                        ]

                    # assistant_message_function_call = delta.delta.function_call

                    # extract tool calls from response
                    if assistant_message_tool_calls:
                        tool_calls = self._extract_response_tool_calls(assistant_message_tool_calls)
                        increase_tool_call(tool_calls)

                    if delta_content is None or delta_content == "":
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(
                        content=delta_content,
                    )

                    # reset tool calls
                    tool_calls = []
                    full_assistant_content += delta_content
                elif "text" in choice:
                    choice_text = choice.get("text", "")
                    if choice_text == "":
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(content=choice_text)
                    full_assistant_content += choice_text
                else:
                    continue

                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=assistant_prompt_message,
                    ),
                )

            chunk_index += 1

        if tools_calls:
            yield LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(tool_calls=tools_calls, content=""),
                ),
            )

        yield create_final_llm_result_chunk(
            id=message_id,
            index=chunk_index,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason or "",
            usage=usage or {},
        )

    def _handle_generate_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        response_json: dict = response.json()

        print(f"response_json: {response_json}")

        completion_type = LLMMode.value_of(credentials["mode"])

        output = response_json["choices"][0]

        response_content = ""
        tool_calls = None
        function_calling_type = credentials.get("function_calling_type", "no_call")
        if completion_type is LLMMode.CHAT:
            response_content = output.get("message", {})["content"]
            if function_calling_type == "tool_call":
                tool_calls = output.get("message", {}).get("tool_calls")
            elif function_calling_type == "function_call":
                tool_calls = output.get("message", {}).get("function_call")

        elif completion_type is LLMMode.COMPLETION:
            response_content = output["text"]

        assistant_message = AssistantPromptMessage(content=response_content, tool_calls=[])

        if tool_calls:
            if function_calling_type == "tool_call":
                assistant_message.tool_calls = self._extract_response_tool_calls(tool_calls)
            elif function_calling_type == "function_call" and tool_calls:
                extracted_tool_call = self._extract_response_function_call(tool_calls)
                if extracted_tool_call:
                    assistant_message.tool_calls = [extracted_tool_call]

        usage = response_json.get("usage")
        if usage:
            # transform usage
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
        else:
            # calculate num tokens
            assert prompt_messages[0].content is not None
            prompt_tokens = self._num_tokens_from_string(model, prompt_messages[0].content)
            assert assistant_message.content is not None
            completion_tokens = self._num_tokens_from_string(model, assistant_message.content)

        # transform usage
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        # transform response
        result = LLMResult(
            model=response_json["model"],
            prompt_messages=prompt_messages,
            message=assistant_message,
            usage=usage,
        )

        return result

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        super().validate_credentials(model, credentials)

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        credentials["mode"] = "chat"
        credentials["endpoint_url"] = "https://maas-api.ai-yuanjing.com/openapi/compatible-mode/v1"

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [],
            InvokeServerUnavailableError: [],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [],
            InvokeBadRequestError: [],
        }