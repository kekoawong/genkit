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

package anthropic

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const provider = "anthropic"

var (
	supportedModels = map[string]ai.ModelInfo{
		"claude-3-7-sonnet-20250219": {
			Label:    "Claude 3.7 Sonnet",
			Supports: compat_oai.Multimodal.Supports,
		},
		"claude-3-5-haiku-20241022": {
			Label:    "Claude 3.5 Haiku",
			Supports: compat_oai.Multimodal.Supports,
		},
		"claude-3-5-sonnet-20240620": {
			Label:    "Claude 3.5 Sonnet",
			Supports: compat_oai.Multimodal.Supports,
		},
		"claude-3-opus-20240229": {
			Label:    "Claude 3 Opus",
			Supports: compat_oai.Multimodal.Supports,
		},
		"claude-3-haiku-20240307": {
			Label:    "Claude 3 Haiku",
			Supports: compat_oai.Multimodal.Supports,
		},
	}
)

type Anthropic struct {
	// TODO: Opts is redundant with openAICompatible.Opts. May be better to embed openAICompatible?
	Opts             []option.RequestOption
	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (a *Anthropic) Name() string {
	return provider
}

func (a *Anthropic) Init(ctx context.Context, g *genkit.Genkit) error {
	// TODO: override options with a.Opts (or embed openAICompatible?)
	a.openAICompatible.Opts = append(a.openAICompatible.Opts, a.Opts...)

	if err := a.openAICompatible.Init(ctx, g); err != nil {
		return err
	}

	// define default models
	for model, info := range supportedModels {
		if _, err := a.DefineModel(g, model, info); err != nil {
			return err
		}
	}

	return nil
}

func (a *Anthropic) Model(g *genkit.Genkit, name string) ai.Model {
	return a.openAICompatible.Model(g, name, provider)
}

func (a *Anthropic) DefineModel(g *genkit.Genkit, name string, info ai.ModelInfo) (ai.Model, error) {
	return a.openAICompatible.DefineModel(g, name, info, provider)
}
