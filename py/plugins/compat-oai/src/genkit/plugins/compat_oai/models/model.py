# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""
OpenAI Compatible Models for Genkit.
"""

import json
from collections.abc import Callable

from openai import OpenAI, pydantic_function_tool
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from genkit.ai.registry import GenkitRegistry
from genkit.core.action import ActionKind
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    Message,
    Role,
    TextPart,
    ToolDefinition,
)


class OpenAIModel:
    """
    Handles OpenAI API interactions for the Genkit plugin.
    """

    def __init__(self, model: str, client: OpenAI, registry: GenkitRegistry):
        """
        Initializes the OpenAIModel instance with the specified model and OpenAI client parameters.

        :param model: The OpenAI model to use for generating responses.
        :param client: OpenAI client instance.
        """
        self._model = model
        self._openai_client = client
        self._registry = registry

    @property
    def name(self):
        return self._model

    def _get_messages(self, messages: list[Message]) -> list[dict]:
        """
        Converts the request messages into the format required by OpenAI's API.

        :param messages: A list of the user messages.
        :return: A list of dictionaries, where each dictionary represents a message
                 with 'role' and 'content' fields.
        :raises ValueError: If no messages are provided in the request.
        """
        if not messages:
            raise ValueError('No messages provided in the request.')
        return [
            {
                'role': m.role.value,
                'content': ''.join(
                    filter(None, (part.root.text for part in m.content))
                ),
            }
            for m in messages
        ]

    def _get_tools_definition(self, tools: list[ToolDefinition]) -> list[dict]:
        """
        Converts the provided tools into OpenAI-compatible function call format.

        :param tools: A list of tool definitions.
        :return: A list of dictionaries representing the formatted tools.
        """
        result = []
        for tool_definition in tools:
            action = self._registry.registry.lookup_action(
                ActionKind.TOOL, tool_definition.name
            )
            function_call = pydantic_function_tool(
                model=action.input_type._type,
                name=tool_definition.name,
                description=tool_definition.description,
            )
            result.append(function_call)
        return result

    def _get_openai_request_config(self, request: GenerateRequest) -> dict:
        openai_config = {
            'messages': self._get_messages(request.messages),
            'model': self._model,
        }
        if request.tools:
            openai_config['tools'] = self._get_tools_definition(request.tools)
        if request.config:
            openai_config.update(**request.config.model_dump())
        return openai_config

    def _evaluate_tool(self, name: str, arguments: str):
        """
        Executes a registered tool with the given arguments and returns the result.

        :param name: Name of the tool to execute.
        :param arguments: JSON-encoded arguments for the tool.
        :return: String representation of the tool's output.
        """
        action = self._registry.registry.lookup_action(ActionKind.TOOL, name)

        # Parse and validate arguments
        arguments = json.loads(arguments)
        arguments = action.input_type.validate_python(arguments)

        # Execute the tool and return its result
        return str(action.run(arguments))

    def _get_evaluated_tool_message_param(
        self, tool_call: ChoiceDeltaToolCall | ChatCompletionMessageToolCall
    ) -> dict:
        """
        Evaluates a tool call and formats its response in a structure compatible with OpenAI's API.

        :param tool_call: The tool call object containing function name and arguments.
        :return: A dictionary formatted as a response message from a tool.
        """
        return {
            'role': Role.TOOL.value,
            'tool_call_id': tool_call.id,
            'content': self._evaluate_tool(
                tool_call.function.name, tool_call.function.arguments
            ),
        }

    def _get_assistant_message_param(
        self, tool_calls: list[ChoiceDeltaToolCall]
    ) -> dict:
        """
        Formats tool call data into assistant message structure compatible with OpenAI's API.

        :param tool_calls: A list of tool call objects.
        :return: A dictionary representing the tool calls formatted for OpenAI.
        """
        return {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': tool_call.id,
                    'type': 'function',
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ],
        }

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Processes the request using OpenAI's chat completion API and returns the generated response.

        :param request: The GenerateRequest object containing the input text and configuration.
        :return: A GenerateResponse object containing the generated message.
        """
        openai_config = self._get_openai_request_config(request=request)
        response = self._openai_client.chat.completions.create(**openai_config)

        while (completion := response.choices[0]).finish_reason == 'tool_calls':
            # Append the assistant's response message
            openai_config['messages'].append(completion.message)

            # Evaluate tool calls and append their responses
            for tool_call in completion.message.tool_calls:
                openai_config['messages'].append(
                    self._get_evaluated_tool_message_param(tool_call)
                )

            response = self._openai_client.chat.completions.create(
                **openai_config
            )

        return GenerateResponse(
            request=request,
            message=Message(
                role=Role.MODEL,
                content=[TextPart(text=completion.message.content)],
            ),
        )

    def generate_stream(
        self, request: GenerateRequest, callback: Callable
    ) -> GenerateResponse:
        """
        Generates a streaming response from the OpenAI client and processes it in chunks.

        Args:
            request (GenerateRequest): The request object containing generation parameters.
            callback (Callable): A function to handle each chunk of the streamed response.

        Returns:
            GenerateResponse: An empty response message when streaming is complete.
        """
        openai_config = self._get_openai_request_config(request=request)
        openai_config['stream'] = True

        # Initiate the streaming response from OpenAI
        stream = self._openai_client.chat.completions.create(**openai_config)

        while not stream.response.is_closed:
            tool_calls = {}

            for chunk in stream:
                choice = chunk.choices[0]

                # Handle standard text content
                if choice.delta.content is not None:
                    callback(
                        GenerateResponseChunk(
                            role=Role.MODEL,
                            index=choice.index,
                            content=[TextPart(text=choice.delta.content)],
                        )
                    )

                # Handle tool calls when OpenAI requires tool execution
                elif choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        # Accumulate fragmented tool call arguments
                        tool_calls.setdefault(
                            tool_call.index, tool_call
                        ).function.arguments += tool_call.function.arguments

            # If tool calls were requested, process them and continue the conversation
            if tool_calls:
                openai_config['messages'].append(
                    self._get_assistant_message_param(tool_calls.values())
                )

                for tool_call in tool_calls.values():
                    openai_config['messages'].append(
                        self._get_evaluated_tool_message_param(tool_call)
                    )

                # Re-initiate streaming after processing tool calls
                stream = self._openai_client.chat.completions.create(
                    **openai_config
                )

        # Return an empty response when streaming is complete
        return GenerateResponse(
            request=request,
            message=Message(role=Role.MODEL, content=[TextPart(text='')]),
        )
