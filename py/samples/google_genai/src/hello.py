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

import asyncio

from pydantic import BaseModel, Field

from genkit.ai import Document, Genkit
from genkit.core.typing import (
    GenerationCommonConfig,
    Message,
    Role,
    TextPart,
)
from genkit.plugins.google_ai.models import gemini
from genkit.plugins.google_genai import (
    EmbeddingTaskType,
    GeminiEmbeddingModels,
    GoogleGenai,
    google_genai_name,
)

ai = Genkit(
    plugins=[GoogleGenai()],
    model=google_genai_name('gemini-2.0-flash'),
)


class GablorkenInput(BaseModel):
    """The Pydantic model for tools."""

    value: int = Field(description='value to calculate gablorken for')


@ai.tool('calculates a gablorken')
def gablorkenTool(input_: GablorkenInput) -> int:
    """The user-defined tool function."""
    return input_.value * 3 - 5


@ai.flow()
async def simple_generate_action_with_tools_flow(value: int) -> str:
    """Generate a greeting for the given name.

    Args:
        value: the integer to send to test function

    Returns:
        The generated response with a function.
    """
    response = await ai.generate(
        model=google_genai_name(gemini.GoogleAiVersion.GEMINI_1_5_FLASH),
        messages=[
            Message(
                role=Role.USER,
                content=[TextPart(text=f'what is a gablorken of {value}')],
            ),
        ],
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def say_hi(data: str):
    resp = await ai.generate(
        prompt=f'hi {data}',
    )
    return resp.text


@ai.flow()
async def embed_docs(docs: list[str]):
    """Generate an embedding for the words in a list.

    Args:
        docs: list of texts (string)

    Returns:
        The generated embedding.
    """
    options = {'task_type': EmbeddingTaskType.CLUSTERING}
    return await ai.embed(
        model=google_genai_name(GeminiEmbeddingModels.TEXT_EMBEDDING_004),
        documents=[Document.from_text(doc) for doc in docs],
        options=options,
    )


@ai.flow()
async def say_hi_with_configured_temperature(data: str):
    return await ai.generate(
        messages=[
            Message(role=Role.USER, content=[TextPart(text=f'hi {data}')])
        ],
        config=GenerationCommonConfig(temperature=0.1),
    )


@ai.flow()
async def say_hi_stream(name: str, ctx):
    stream, _ = ai.generate_stream(
        prompt=f'hi {name}',
    )
    result = ''
    async for data in stream:
        ctx.send_chunk(data.text)
        for part in data.content:
            result += part.root.text

    return result


async def main() -> None:
    print(await say_hi(', tell me a joke'))


if __name__ == '__main__':
    asyncio.run(main())


# prevent app from exiting when genkit is running in dev mode
ai.join()
