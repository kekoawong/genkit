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

"""Action module for defining and managing RPC-over-HTTP functions.

This module provides the core functionality for creating and managing actions in
the Genkit framework. Actions are strongly-typed, named, observable,
uninterrupted operations that can operate in streaming or non-streaming mode.
"""

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from enum import StrEnum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from genkit.aio import Channel
from genkit.codec import dump_json
from genkit.core.error import GenkitError
from genkit.core.tracing import tracer

# TODO: add typing, generics
StreamingCallback = Callable[[Any], None]


_action_context: ContextVar[dict[str, Any] | None] = ContextVar('context')
_action_context.set(None)


class ActionKind(StrEnum):
    """Enumerates all the types of action that can be registered.

    This enum defines the various types of actions supported by the framework,
    including chat models, embedders, evaluators, and other utility functions.
    """

    CUSTOM = 'custom'
    EMBEDDER = 'embedder'
    EVALUATOR = 'evaluator'
    EXECUTABLE_PROMPT = 'executable-prompt'
    FLOW = 'flow'
    INDEXER = 'indexer'
    MODEL = 'model'
    PROMPT = 'prompt'
    RERANKER = 'reranker'
    RETRIEVER = 'retriever'
    TOOL = 'tool'
    UTIL = 'util'


class ActionResponse(BaseModel):
    """The response from an action.

    Attributes:
        response: The actual response data from the action execution.
        trace_id: A unique identifier for tracing the action execution.
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    response: Any
    trace_id: str = Field(alias='traceId')


class ActionMetadataKey(StrEnum):
    """Enumerates all the keys of the action metadata.

    Attributes:
        INPUT_KEY: Key for the input schema metadata.
        OUTPUT_KEY: Key for the output schema metadata.
        RETURN: Key for the return type metadata.
    """

    INPUT_KEY = 'inputSchema'
    OUTPUT_KEY = 'outputSchema'
    RETURN = 'return'


def parse_action_key(key: str) -> tuple[ActionKind, str]:
    """Parse an action key into its kind and name components.

    Args:
        key: The action key to parse, in the format "/kind/name".

    Returns:
        A tuple containing the ActionKind and name.

    Raises:
        ValueError: If the key format is invalid or if the kind is not a valid
            ActionKind.
    """
    tokens = key.split('/')
    if len(tokens) < 3 or not tokens[1] or not tokens[2]:
        msg = (
            f'Invalid action key format: `{key}`.'
            'Expected format: `/<kind>/<name>`'
        )
        raise ValueError(msg)

    kind_str = tokens[1]
    name = '/'.join(tokens[2:])
    try:
        kind = ActionKind(kind_str)
    except ValueError as e:
        msg = f'Invalid action kind: `{kind_str}`'
        raise ValueError(msg) from e
    return kind, name


def parse_plugin_name_from_action_name(name: str) -> str | None:
    """Parses the plugin name from an action name.

    As per convention, the plugin name is optional. If present, it's the first
    part of the action name, separated by a forward slash: `pluginname/*`.

    Args:
        name: The action name string.

    Returns:
        The plugin name, or None if no plugin name is found in the action name.
    """
    tokens = name.split('/')
    if len(tokens) > 1:
        return tokens[0]
    return None


def create_action_key(kind: ActionKind, name: str) -> str:
    """Create an action key from its kind and name components.

    Args:
        kind: The kind of action.
        name: The name of the action.

    Returns:
        The action key in the format `/<kind>/<name>`.
    """
    return f'/{kind}/{name}'


def noop_streaming_callback(chunk: Any) -> None:
    """A no-op streaming callback.

    This callback does nothing and is used when no streaming is desired.
    """
    pass


class ActionRunContext:
    """Context for an action execution."""

    def __init__(
        self,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize an ActionRunContext.

        Args:
            on_chunk: The callback to invoke when a chunk is received.
            context: The context to pass to the action.
        """
        self._on_chunk = (
            on_chunk if on_chunk is not None else noop_streaming_callback
        )
        self.context = context if context is not None else {}

    @cached_property
    def is_streaming(self) -> bool:
        """Determines whether context contains on chunk callback.

        Returns:
            Boolean indicating whether the context contains a streaming
            callback.
        """
        return self._on_chunk != noop_streaming_callback

    def send_chunk(self, chunk: Any) -> None:
        """Send a chunk to from the action to the client.

        Args:
            chunk: The chunk to send to the client.
        """
        self._on_chunk(chunk)

    @staticmethod
    def _current_context() -> dict[str, Any] | None:
        """Obtains current context if running within an action.

        Returns:
            The current context if running within an action, None otherwise.
        """
        return _action_context.get(None)


class Action:
    """An action is a Typed JSON-based RPC-over-HTTP remote-callable function.

    Actions support metadata, streaming, reflection and discovery. They are
    strongly-typed, named, observable, uninterrupted operations that can operate
    in streaming or non-streaming mode. An action wraps a function that takes an
    input and returns an output, optionally streaming values incrementally by
    invoking a streaming callback.
    """

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        fn: Callable[..., Any],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        span_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an Action.

        Args:
            kind: The kind of action (e.g., TOOL, MODEL, etc.).
            name: Unique name identifier for this action.
            fn: The function to call when the action is executed.
            description: Optional human-readable description of the action.
            metadata: Optional dictionary of metadata about the action.
            span_metadata: Optional dictionary of tracing span metadata.
        """
        self.kind = kind
        self.name = name

        input_spec = inspect.getfullargspec(fn)
        arg_types = []

        action_args = input_spec.args.copy()

        # Special case when using a method as an action, we ignore first "self"
        # arg.
        if (
            len(action_args) > 0
            and len(action_args) <= 3
            and action_args[0] == 'self'
        ):
            del action_args[0]

        for arg in action_args:
            arg_types.append(
                input_spec.annotations[arg]
                if arg in input_spec.annotations
                else Any
            )

        afn = ensure_async(fn)
        self.is_async = asyncio.iscoroutinefunction(fn)

        async def async_tracing_wrapper(
            input: Any | None, ctx: ActionRunContext
        ) -> ActionResponse:
            """Wrap the function in an async tracing wrapper.

            Args:
                input: The input to the action.
                ctx: The context to pass to the action.

            Returns:
                The action response.
            """
            with tracer.start_as_current_span(name) as span:
                trace_id = str(span.get_span_context().trace_id)
                record_input_metadata(
                    span=span,
                    kind=kind,
                    name=name,
                    span_metadata=span_metadata,
                    input=input,
                )

                try:
                    match len(action_args):
                        case 0:
                            output = await afn()
                        case 1:
                            output = await afn(input)
                        case 2:
                            output = await afn(input, ctx)
                        case _:
                            raise ValueError('action fn must have 0-2 args...')
                except Exception as e:
                    raise GenkitError(
                        cause=e.cause
                        if isinstance(e, GenkitError) and e.cause
                        else e,
                        message=f'Error while running action {self.name}',
                        trace_id=trace_id,
                    )

                record_output_metadata(span, output=output)
                return ActionResponse(response=output, trace_id=trace_id)

        def sync_tracing_wrapper(
            input: Any | None, ctx: ActionRunContext
        ) -> ActionResponse:
            """Wrap the function in a sync tracing wrapper.

            Args:
                input: The input to the action.
                ctx: The context to pass to the action.

            Returns:
                The action response.
            """
            with tracer.start_as_current_span(name) as span:
                trace_id = str(span.get_span_context().trace_id)
                record_input_metadata(
                    span=span,
                    kind=kind,
                    name=name,
                    span_metadata=span_metadata,
                    input=input,
                )

                try:
                    match len(action_args):
                        case 0:
                            output = fn()
                        case 1:
                            output = fn(input)
                        case 2:
                            output = fn(input, ctx)
                        case _:
                            raise ValueError('action fn must have 0-2 args...')
                except Exception as e:
                    raise GenkitError(
                        cause=e,
                        message=f'Error while running action {self.name}',
                        trace_id=trace_id,
                    )

                record_output_metadata(span, output=output)
                return ActionResponse(response=output, trace_id=trace_id)

        self.__fn = sync_tracing_wrapper
        self.__afn = async_tracing_wrapper
        self.description = description
        self.metadata = metadata if metadata else {}

        if len(action_args) > 2:
            raise Exception(f'can only have up to 2 arg: {action_args}')
        if len(action_args) > 0:
            type_adapter = TypeAdapter(arg_types[0])
            self.input_schema = type_adapter.json_schema()
            self.input_type = type_adapter
            self.metadata[ActionMetadataKey.INPUT_KEY] = self.input_schema
        else:
            self.input_schema = TypeAdapter(Any).json_schema()
            self.input_type = None
            self.metadata[ActionMetadataKey.INPUT_KEY] = self.input_schema

        if ActionMetadataKey.RETURN in input_spec.annotations:
            type_adapter = TypeAdapter(
                input_spec.annotations[ActionMetadataKey.RETURN]
            )
            self.output_schema = type_adapter.json_schema()
            self.metadata[ActionMetadataKey.OUTPUT_KEY] = self.output_schema
        else:
            self.output_schema = TypeAdapter(Any).json_schema()
            self.metadata[ActionMetadataKey.OUTPUT_KEY] = self.output_schema

    def run(
        self,
        input: Any = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ) -> ActionResponse:
        """Run the action with input.

        Args:
            input: The input to the action.
            on_chunk: The callback to invoke when a chunk is received.
            context: The context to pass to the action.
            telemetry_labels: The telemetry labels to pass to the action.

        Returns:
            The action response.
        """
        # TODO: handle telemetry_labels

        if context:
            _action_context.set(context)

        return self.__fn(
            input,
            ActionRunContext(
                on_chunk=on_chunk, context=_action_context.get(None)
            ),
        )

    async def arun(
        self,
        input: Any = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ) -> ActionResponse:
        """Run the action with raw input.

        Args:
            input: The input to the action.
            on_chunk: The callback to invoke when a chunk is received.
            context: The context to pass to the action.
            telemetry_labels: The telemetry labels to pass to the action.

        Returns:
            The action response.
        """
        # TODO: handle telemetry_labels

        if context:
            _action_context.set(context)

        return await self.__afn(
            input,
            ActionRunContext(
                on_chunk=on_chunk, context=_action_context.get(None)
            ),
        )

    async def arun_raw(
        self,
        raw_input: Any,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ):
        """Run the action with raw input.

        Args:
            raw_input: The raw input to the action.
            on_chunk: The callback to invoke when a chunk is received.
            context: The context to pass to the action.
            telemetry_labels: The telemetry labels to pass to the action.

        Returns:
            The action response.
        """
        input_action = (
            self.input_type.validate_python(raw_input)
            if self.input_type is not None
            else None
        )
        return await self.arun(
            input=input_action,
            on_chunk=on_chunk,
            context=context,
            telemetry_labels=telemetry_labels,
        )

    def stream(
        self,
        input: Any = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ) -> tuple[
        AsyncIterator[ActionResponse],
        asyncio.Future[ActionResponse],
    ]:
        """Run the action and return an async iterator of the results.

        Args:
            input: The input to the action.
            context: The context to pass to the action.
            telemetry_labels: The telemetry labels to pass to the action.

        Returns:
            A tuple containing:
            - An AsyncIterator of the chunks from the action.
            - An asyncio.Future that resolves to the final result of the action.
        """
        stream = Channel()

        resp = self.arun(
            input=input,
            context=context,
            telemetry_labels=telemetry_labels,
            on_chunk=lambda c: stream.send(c),
        )
        stream.set_close_future(resp)

        result_future: asyncio.Future[ActionResponse] = asyncio.Future()
        stream.closed.add_done_callback(
            lambda _: result_future.set_result(stream.closed.result().response)
        )

        return (stream, result_future)


def record_input_metadata(span, kind, name, span_metadata, input):
    """Record the input metadata for the action.

    Args:
        span: The span to record the metadata for.
        kind: The kind of action.
        name: The name of the action.
        span_metadata: The span metadata to record.
        input: The input to the action.
    """
    span.set_attribute('genkit:type', 'action')
    span.set_attribute('genkit:metadata:subtype', kind)
    span.set_attribute('genkit:name', name)
    if input is not None:
        span.set_attribute('genkit:input', dump_json(input))

    if span_metadata is not None:
        for meta_key in span_metadata:
            span.set_attribute(meta_key, span_metadata[meta_key])


def record_output_metadata(span, output) -> None:
    """Record the output metadata for the action.

    Args:
        span: The span to record the metadata for.
        output: The output to the action.
    """
    span.set_attribute('genkit:state', 'success')
    span.set_attribute('genkit:output', dump_json(output))


def ensure_async(fn: Callable) -> Callable:
    """Ensure the function is async.

    Args:
        fn: The function to ensure is async.

    Returns:
        The async function.
    """
    is_async = asyncio.iscoroutinefunction(fn)
    if is_async:
        return fn

    async def async_wrapper(*args, **kwargs):
        """Wrap the function in an async function.

        Args:
            *args: The arguments to the function.
            **kwargs: The keyword arguments to the function.

        Returns:
            The result of the function.
        """
        return fn(*args, **kwargs)

    return async_wrapper
