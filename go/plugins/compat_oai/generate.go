// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compat_oai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

// ModelGenerator handles OpenAI generation requests
type ModelGenerator struct {
	client    *openai.Client
	modelName string
	request   *openai.ChatCompletionNewParams
}

// NewModelGenerator creates a new ModelGenerator instance
func NewModelGenerator(client *openai.Client, modelName string) *ModelGenerator {
	return &ModelGenerator{
		client:    client,
		modelName: modelName,
		request: &openai.ChatCompletionNewParams{
			Model: openai.F(modelName),
		},
	}
}

// WithMessages adds messages to the request
func (g *ModelGenerator) WithMessages(messages []*ai.Message) *ModelGenerator {
	oaiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		content := g.concatenateContent(msg.Content)
		switch msg.Role {
		case ai.RoleSystem:
			oaiMessages = append(oaiMessages, openai.SystemMessage(content))
		case ai.RoleModel:
			oaiMessages = append(oaiMessages, openai.AssistantMessage(content))

			am := openai.ChatCompletionAssistantMessageParam{
				Role: openai.F(openai.ChatCompletionAssistantMessageParamRoleAssistant),
			}
			if msg.Content[0].Text != "" {
				am.Content = openai.F([]openai.ChatCompletionAssistantMessageParamContentUnion{
					openai.TextPart(msg.Content[0].Text),
				})
			}
			toolCalls := convertToolCalls(msg.Content)
			if len(toolCalls) > 0 {
				am.ToolCalls = openai.F(toolCalls)
			}
			oaiMessages = append(oaiMessages, am)
		case ai.RoleTool:
			for _, p := range msg.Content {
				if !p.IsToolResponse() {
					continue
				}
				tm := openai.ToolMessage(
					// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolResponse struct.
					p.ToolResponse.Name,
					anyToJSONString(p.ToolResponse.Output),
				)
				oaiMessages = append(oaiMessages, tm)
			}
		default:
			oaiMessages = append(oaiMessages, openai.UserMessage(content))
		}
	}
	g.request.Messages = openai.F(oaiMessages)
	return g
}

// WithConfig adds configuration parameters from the model request
func (g *ModelGenerator) WithConfig(config any) *ModelGenerator {
	if config != nil {
		// TODO: Implement configuration from model request
		// modelRequest.Config is any type
	}
	return g
}

// WithTools adds tools to the request
func (g *ModelGenerator) WithTools(tools []*ai.ToolDefinition, choice ai.ToolChoice) *ModelGenerator {
	if len(tools) == 0 {
		return g
	}

	toolParams := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		if tool == nil || tool.Name == "" {
			continue
		}

		toolParams = append(toolParams, openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(shared.FunctionDefinitionParam{
				Name:        openai.F(tool.Name),
				Description: openai.F(tool.Description),
				Parameters:  openai.F(openai.FunctionParameters(tool.InputSchema)),
				Strict:      openai.F(false), // TODO: implement strict mode
			}),
		})
	}
	g.request.Tools = openai.F(toolParams)

	switch choice {
	case ai.ToolChoiceAuto:
		g.request.ToolChoice = openai.F[openai.ChatCompletionToolChoiceOptionUnionParam](openai.ChatCompletionToolChoiceOptionAutoAuto)
	case ai.ToolChoiceRequired:
		g.request.ToolChoice = openai.F[openai.ChatCompletionToolChoiceOptionUnionParam](openai.ChatCompletionToolChoiceOptionAutoRequired)
	case ai.ToolChoiceNone:
		g.request.ToolChoice = openai.F[openai.ChatCompletionToolChoiceOptionUnionParam](openai.ChatCompletionToolChoiceOptionAutoNone)
	}

	return g
}

// Generate executes the generation request
func (g *ModelGenerator) Generate(ctx context.Context, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	if handleChunk != nil {
		return g.generateStream(ctx, handleChunk)
	}
	return g.generateComplete(ctx)
}

// concatenateContent concatenates text content into a single string
func (g *ModelGenerator) concatenateContent(parts []*ai.Part) string {
	content := ""
	for _, part := range parts {
		content += part.Text
	}
	return content
}

// generateStream generates a streaming model response
// TODO: Implement tool calls
func (g *ModelGenerator) generateStream(ctx context.Context, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	stream := g.client.Chat.Completions.NewStreaming(ctx, *g.request)
	defer stream.Close()

	var fullResponse ai.ModelResponse
	fullResponse.Message = &ai.Message{
		Role:    ai.RoleModel,
		Content: make([]*ai.Part, 0),
	}

	// Initialize request and usage
	fullResponse.Request = &ai.ModelRequest{}
	fullResponse.Usage = &ai.GenerationUsage{
		InputTokens:  0,
		OutputTokens: 0,
		TotalTokens:  0,
	}

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			modelChunk := &ai.ModelResponseChunk{
				Content: []*ai.Part{ai.NewTextPart(content)},
			}

			if err := handleChunk(ctx, modelChunk); err != nil {
				return nil, fmt.Errorf("callback error: %w", err)
			}

			fullResponse.Message.Content = append(fullResponse.Message.Content, modelChunk.Content...)

			// Update Usage
			fullResponse.Usage.InputTokens += int(chunk.Usage.PromptTokens)
			fullResponse.Usage.OutputTokens += int(chunk.Usage.CompletionTokens)
			fullResponse.Usage.TotalTokens += int(chunk.Usage.TotalTokens)
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	return &fullResponse, nil
}

// generateComplete generates a complete model response
func (g *ModelGenerator) generateComplete(ctx context.Context) (*ai.ModelResponse, error) {
	completion, err := g.client.Chat.Completions.New(ctx, *g.request)
	if err != nil {
		return nil, fmt.Errorf("failed to create completion: %w", err)
	}

	choice := completion.Choices[0]

	resp := &ai.ModelResponse{
		Request: &ai.ModelRequest{},
		Usage: &ai.GenerationUsage{
			InputTokens:  int(completion.Usage.PromptTokens),
			OutputTokens: int(completion.Usage.CompletionTokens),
			TotalTokens:  int(completion.Usage.TotalTokens),
		},
	}

	switch choice.FinishReason {
	case openai.ChatCompletionChoicesFinishReasonStop, openai.ChatCompletionChoicesFinishReasonToolCalls:
		resp.FinishReason = ai.FinishReasonStop
	case openai.ChatCompletionChoicesFinishReasonLength:
		resp.FinishReason = ai.FinishReasonLength
	case openai.ChatCompletionChoicesFinishReasonContentFilter:
		resp.FinishReason = ai.FinishReasonBlocked
	case openai.ChatCompletionChoicesFinishReasonFunctionCall:
		resp.FinishReason = ai.FinishReasonOther
	default:
		resp.FinishReason = ai.FinishReasonUnknown
	}

	message := &ai.Message{
		Role: ai.RoleModel,
	}

	// TODO: Implement tool calls
	// handle tool calls
	var toolRequestParts []*ai.Part
	for _, toolCall := range choice.Message.ToolCalls {
		toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  toolCall.Function.Name,
			Input: jsonStringToMap(toolCall.Function.Arguments),
		}))
	}
	if len(toolRequestParts) > 0 {
		message.Content = toolRequestParts
		resp.Message = message
		return resp, nil
	}

	message.Content = []*ai.Part{
		ai.NewTextPart(completion.Choices[0].Message.Content),
	}
	resp.Message = message
	return resp, nil
}

func convertToolCalls(content []*ai.Part) []openai.ChatCompletionMessageToolCallParam {
	var toolCalls []openai.ChatCompletionMessageToolCallParam
	for _, p := range content {
		if !p.IsToolRequest() {
			continue
		}
		toolCall := convertToolCall(p)
		toolCalls = append(toolCalls, toolCall)
	}
	return toolCalls
}

func convertToolCall(part *ai.Part) openai.ChatCompletionMessageToolCallParam {
	param := openai.ChatCompletionMessageToolCallParam{
		// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolRequest struct.
		ID:   openai.F(part.ToolRequest.Name),
		Type: openai.F(openai.ChatCompletionMessageToolCallTypeFunction),
		Function: openai.F(openai.ChatCompletionMessageToolCallFunctionParam{
			Name: openai.F(part.ToolRequest.Name),
		}),
	}

	if part.ToolRequest.Input != nil {
		param.Function.Value.Arguments = openai.F(anyToJSONString(part.ToolRequest.Input))
	}

	return param
}

func jsonStringToMap(jsonString string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		panic(fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonString, err))
	}
	return result
}

func anyToJSONString(data any) string {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		panic(fmt.Errorf("failed to marshal any to JSON string: data, %#v %w", data, err))
	}
	return string(jsonBytes)
}
