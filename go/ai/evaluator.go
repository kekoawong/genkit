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
//
// SPDX-License-Identifier: Apache-2.0

package ai

import (
	"context"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/atype"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
)

// Evaluator represents a evaluator action.
type Evaluator interface {
	// Name returns the name of the evaluator.
	Name() string
	// Evaluates a dataset.
	Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error)
}

type (
	evaluatorActionDef core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}]

	evaluatorAction = core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}]
)

// Example is a single example that requires evaluation
type Example struct {
	TestCaseId string   `json:"testCaseId,omitempty"`
	Input      any      `json:"input"`
	Output     any      `json:"output,omitempty"`
	Context    []any    `json:"context,omitempty"`
	Reference  any      `json:"reference,omitempty"`
	TraceIds   []string `json:"traceIds,omitempty"`
}

// Dataset is a collection of [Example]
type Dataset = []Example

// EvaluatorRequest is the data we pass to evaluate a dataset.
// The Options field is specific to the actual evaluator implementation.
type EvaluatorRequest struct {
	Dataset      *Dataset `json:"dataset"`
	EvaluationId string   `json:"evalRunId"`
	Options      any      `json:"options,omitempty"`
}

// ScoreStatus is an enum used to indicate if a Score has passed or failed. This
// drives additional features in tooling / the Dev UI.
type ScoreStatus int

const (
	ScoreStatusUnknown ScoreStatus = iota
	ScoreStatusFail
	ScoreStatusPass
)

var statusName = map[ScoreStatus]string{
	ScoreStatusUnknown: "unknown",
	ScoreStatusFail:    "fail",
	ScoreStatusPass:    "pass",
}

func (ss ScoreStatus) String() string {
	return statusName[ss]
}

// Score is the evaluation score that represents the result of an evaluator.
// This struct includes information such as the score (numeric, string or other
// types), the reasoning provided for this score (if any), the score status (if
// any) and other details.
type Score struct {
	Id      string         `json:"id,omitempty"`
	Score   any            `json:"score,omitempty"`
	Status  string         `json:"status,omitempty" jsonschema:"enum=unknown,enum=fail,enum=pass"`
	Error   string         `json:"error,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}

// EvaluationResult is the result of running the evaluator on a single Example.
// An evaluator may provide multiple scores simultaneously (e.g. if they are using
// an API to score on multiple criteria)
type EvaluationResult struct {
	TestCaseId string  `json:"testCaseId"`
	TraceID    string  `json:"traceId,omitempty"`
	SpanID     string  `json:"spanId,omitempty"`
	Evaluation []Score `json:"evaluation"`
}

// EvaluatorResponse is a collection of [EvaluationResult] structs, it
// represents the result on the entire input dataset.
type EvaluatorResponse = []EvaluationResult

type EvaluatorOptions struct {
	DisplayName string `json:"displayName"`
	Definition  string `json:"definition"`
	IsBilled    bool   `json:"isBilled,omitempty"`
}

// EvaluatorCallbackRequest is the data we pass to the callback function
// provided in defineEvaluator. The Options field is specific to the actual
// evaluator implementation.
type EvaluatorCallbackRequest struct {
	Input   Example `json:"input"`
	Options any     `json:"options,omitempty"`
}

// EvaluatorCallbackResponse is the result on evaluating a single [Example]
type EvaluatorCallbackResponse = EvaluationResult

// DefineEvaluator registers the given evaluator function as an action, and
// returns a [Evaluator] that runs it. This method process the input dataset
// one-by-one.
func DefineEvaluator(r *registry.Registry, provider, name string, options *EvaluatorOptions, eval func(context.Context, *EvaluatorCallbackRequest) (*EvaluatorCallbackResponse, error)) (Evaluator, error) {
	if options == nil {
		return nil, errors.New("EvaluatorOptions must be provided")
	}
	// TODO(ssbushi): Set this on `evaluator` key on action metadata
	metadataMap := map[string]any{}
	metadataMap["evaluatorIsBilled"] = options.IsBilled
	metadataMap["evaluatorDisplayName"] = options.DisplayName
	metadataMap["evaluatorDefinition"] = options.Definition

	actionDef := (*evaluatorActionDef)(core.DefineAction(r, provider, name, atype.Evaluator, metadataMap, func(ctx context.Context, req *EvaluatorRequest) (output *EvaluatorResponse, err error) {
		var evalResponses []EvaluationResult
		dataset := *req.Dataset
		for i := 0; i < len(dataset); i++ {
			datapoint := dataset[i]
			if datapoint.TestCaseId == "" {
				datapoint.TestCaseId = uuid.New().String()
			}
			_, err := tracing.RunInNewSpan(ctx, r.TracingState(), fmt.Sprintf("TestCase %s", datapoint.TestCaseId), "evaluator", false, datapoint,
				func(ctx context.Context, input Example) (*EvaluatorCallbackResponse, error) {
					traceId := trace.SpanContextFromContext(ctx).TraceID().String()
					spanId := trace.SpanContextFromContext(ctx).SpanID().String()
					callbackRequest := EvaluatorCallbackRequest{
						Input:   input,
						Options: req.Options,
					}
					evaluatorResponse, err := eval(ctx, &callbackRequest)
					if err != nil {
						failedScore := Score{
							Status: ScoreStatusFail.String(),
							Error:  fmt.Sprintf("Evaluation of test case %s failed: \n %s", input.TestCaseId, err.Error()),
						}
						failedEvalResult := EvaluationResult{
							TestCaseId: input.TestCaseId,
							Evaluation: []Score{failedScore},
							TraceID:    traceId,
							SpanID:     spanId,
						}
						evalResponses = append(evalResponses, failedEvalResult)
						// return error to mark span as failed
						return nil, err
					}
					evaluatorResponse.TraceID = traceId
					evaluatorResponse.SpanID = spanId
					evalResponses = append(evalResponses, *evaluatorResponse)
					return evaluatorResponse, nil
				})
			if err != nil {
				logger.FromContext(ctx).Debug("EvaluatorAction", "err", err)
				continue
			}
		}
		return &evalResponses, nil
	}))
	return actionDef, nil
}

// DefineBatchEvaluator registers the given evaluator function as an action, and
// returns a [Evaluator] that runs it. This method provide the full
// [EvaluatorRequest] to the callback function, giving more flexibilty to the
// user for processing the data, such as batching or parallelization.
func DefineBatchEvaluator(r *registry.Registry, provider, name string, options *EvaluatorOptions, batchEval func(context.Context, *EvaluatorRequest) (*EvaluatorResponse, error)) (Evaluator, error) {
	if options == nil {
		return nil, errors.New("EvaluatorOptions must be provided")
	}

	metadataMap := map[string]any{}
	metadataMap["evaluatorIsBilled"] = options.IsBilled
	metadataMap["evaluatorDisplayName"] = options.DisplayName
	metadataMap["evaluatorDefinition"] = options.Definition

	return (*evaluatorActionDef)(core.DefineAction(r, provider, name, atype.Evaluator, map[string]any{"evaluator": metadataMap}, batchEval)), nil
}

// IsDefinedEvaluator reports whether an [Evaluator] is defined.
func IsDefinedEvaluator(r *registry.Registry, provider, name string) bool {
	return (*evaluatorActionDef)(core.LookupActionFor[*EvaluatorRequest, *EvaluatorResponse, struct{}](r, atype.Evaluator, provider, name)) != nil
}

// LookupEvaluator looks up an [Evaluator] registered by [DefineEvaluator].
// It returns nil if the evaluator was not defined.
func LookupEvaluator(r *registry.Registry, provider, name string) Evaluator {
	return (*evaluatorActionDef)(core.LookupActionFor[*EvaluatorRequest, *EvaluatorResponse, struct{}](r, atype.Evaluator, provider, name))
}

// EvaluateOption configures params of the Embed call.
type EvaluateOption func(req *EvaluatorRequest) error

// WithEvaluateDataset set the dataset on [EvaluatorRequest]
func WithEvaluateDataset(dataset *Dataset) EvaluateOption {
	return func(req *EvaluatorRequest) error {
		req.Dataset = dataset
		return nil
	}
}

// WithEvaluateId set evaluation ID on [EvaluatorRequest]
func WithEvaluateId(evaluationId string) EvaluateOption {
	return func(req *EvaluatorRequest) error {
		req.EvaluationId = evaluationId
		return nil
	}
}

// WithEvaluateOptions set evaluator options on [EvaluatorRequest]
func WithEvaluateOptions(opts any) EvaluateOption {
	return func(req *EvaluatorRequest) error {
		req.Options = opts
		return nil
	}
}

// Evaluate calls the retrivers with provided options.
func Evaluate(ctx context.Context, r Evaluator, opts ...EvaluateOption) (*EvaluatorResponse, error) {
	req := &EvaluatorRequest{}
	for _, with := range opts {
		err := with(req)
		if err != nil {
			return nil, err
		}
	}
	return r.Evaluate(ctx, req)
}

func (r *evaluatorActionDef) Name() string { return (*evaluatorAction)(r).Name() }

// Evaluate runs the given [Evaluator].
func (e *evaluatorActionDef) Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error) {
	if e == nil {
		return nil, errors.New("Evaluator called on a nil Evaluator; check that all evaluators are defined")
	}
	a := (*core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}])(e)
	return a.Run(ctx, req, nil)
}
