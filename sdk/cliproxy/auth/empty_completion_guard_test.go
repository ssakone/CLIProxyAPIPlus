package auth

import (
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
)

// These tests guard two behaviors that regressed on the live branch when its
// empty-completion detector misjudged codex /v1/responses results:
//   - a Responses payload whose shape cannot be inspected (for example "text"
//     carrying the format configuration object) must be forwarded, never
//     turned into an empty-completion failure;
//   - a locally detected empty result must not put the credential into a
//     timed cooldown, otherwise a single-auth deployment rejects all traffic
//     for that model with auth_unavailable.

const emptyGuardProvider = "codex"

type emptyGuardExecutor struct {
	mu            sync.Mutex
	payload       []byte
	emptyStream   bool
	executeCalls  int
	streamCalls   int
	streamPayload []byte
}

func (e *emptyGuardExecutor) Identifier() string { return emptyGuardProvider }

func (e *emptyGuardExecutor) Execute(_ context.Context, _ *Auth, _ cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.executeCalls++
	return cliproxyexecutor.Response{Payload: append([]byte(nil), e.payload...)}, nil
}

func (e *emptyGuardExecutor) ExecuteStream(_ context.Context, _ *Auth, _ cliproxyexecutor.Request, _ cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	e.mu.Lock()
	e.streamCalls++
	empty := e.emptyStream
	payload := append([]byte(nil), e.streamPayload...)
	e.mu.Unlock()
	ch := make(chan cliproxyexecutor.StreamChunk, 1)
	if !empty {
		ch <- cliproxyexecutor.StreamChunk{Payload: payload}
	}
	close(ch)
	return &cliproxyexecutor.StreamResult{Headers: http.Header{}, Chunks: ch}, nil
}

func (e *emptyGuardExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) { return auth, nil }

func (e *emptyGuardExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, nil
}

func (e *emptyGuardExecutor) HttpRequest(context.Context, *Auth, *http.Request) (*http.Response, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "HttpRequest not implemented"}
}

func newEmptyGuardManager(t *testing.T, model string, executor *emptyGuardExecutor) (*Manager, *Auth) {
	t.Helper()
	prevDisabled := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(prevDisabled) })
	prevTransient := transientErrorCooldownSeconds.Load()
	SetTransientErrorCooldownSeconds(60)
	t.Cleanup(func() { transientErrorCooldownSeconds.Store(prevTransient) })

	m := NewManager(nil, nil, nil)
	m.RegisterExecutor(executor)
	auth := &Auth{ID: "empty-guard-" + t.Name(), Provider: emptyGuardProvider, Status: StatusActive}
	if _, errRegister := m.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("register auth: %v", errRegister)
	}
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth.ID, emptyGuardProvider, []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() { reg.UnregisterClient(auth.ID) })
	return m, auth
}

func assertAuthNotCoolingDown(t *testing.T, m *Manager, authID, model string) {
	t.Helper()
	updated, ok := m.GetByID(authID)
	if !ok || updated == nil {
		t.Fatal("expected auth to be present")
	}
	if blocked, reason, next := isAuthBlockedForModel(updated, model, time.Now()); blocked {
		t.Fatalf("auth blocked for %s (reason=%v, next=%v), want selectable", model, reason, next)
	}
}

func TestManagerExecute_UninspectableResponsesPayloadIsForwarded(t *testing.T) {
	const model = "gpt-6-luna"
	for name, payload := range map[string]string{
		"text format configuration": `{"object":"response","status":"completed","text":{"format":{"type":"text"},"verbosity":"medium"},"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"OK","annotations":[]}]}],"usage":{"input_tokens":9,"output_tokens":5,"total_tokens":14}}`,
		"non-object usage":          `{"object":"response","usage":[]}`,
		"object delta":              `{"type":"response.output_text.delta","delta":{"text":"OK"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			executor := &emptyGuardExecutor{payload: []byte(payload), streamPayload: []byte("data: " + payload + "\n\n")}
			m, auth := newEmptyGuardManager(t, model, executor)

			resp, errExec := m.Execute(context.Background(), []string{emptyGuardProvider}, cliproxyexecutor.Request{Model: model}, cliproxyexecutor.Options{})
			if errExec != nil {
				t.Fatalf("Execute error = %v, want payload forwarded", errExec)
			}
			if string(resp.Payload) != payload {
				t.Fatalf("Execute payload = %s, want %s", resp.Payload, payload)
			}

			stream, errStream := m.ExecuteStream(context.Background(), []string{emptyGuardProvider}, cliproxyexecutor.Request{Model: model}, cliproxyexecutor.Options{Stream: true})
			if errStream != nil {
				t.Fatalf("ExecuteStream error = %v, want payload forwarded", errStream)
			}
			var got []byte
			for chunk := range stream.Chunks {
				if chunk.Err != nil {
					t.Fatalf("stream chunk error: %v", chunk.Err)
				}
				got = append(got, chunk.Payload...)
			}
			if string(got) != string(executor.streamPayload) {
				t.Fatalf("stream payload = %q, want %q", got, executor.streamPayload)
			}
			assertAuthNotCoolingDown(t, m, auth.ID, model)
		})
	}
}

func TestManagerExecuteStream_EmptyResultDoesNotCooldownCredential(t *testing.T) {
	const model = "gpt-6-luna"
	executor := &emptyGuardExecutor{emptyStream: true, streamPayload: []byte("data: {\"type\":\"response.completed\"}\n\n")}
	m, auth := newEmptyGuardManager(t, model, executor)

	first, errFirst := m.ExecuteStream(context.Background(), []string{emptyGuardProvider}, cliproxyexecutor.Request{Model: model}, cliproxyexecutor.Options{Stream: true})
	if errFirst != nil {
		t.Fatalf("first ExecuteStream error = %v, want bootstrap error chunk", errFirst)
	}
	var firstErr error
	for chunk := range first.Chunks {
		if chunk.Err != nil {
			firstErr = chunk.Err
		}
	}
	if firstErr == nil {
		t.Fatal("first stream carried no error, want empty_stream failure")
	}
	assertAuthNotCoolingDown(t, m, auth.ID, model)

	executor.mu.Lock()
	executor.emptyStream = false
	executor.mu.Unlock()

	// The only credential must still serve the next request.
	stream, errSecond := m.ExecuteStream(context.Background(), []string{emptyGuardProvider}, cliproxyexecutor.Request{Model: model}, cliproxyexecutor.Options{Stream: true})
	if errSecond != nil {
		t.Fatalf("second ExecuteStream error = %v, want credential still selectable", errSecond)
	}
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}
}

func TestManagerMarkResult_EmptyVerdictDoesNotCooldownButUpstream503Does(t *testing.T) {
	const model = "gpt-6-luna"

	t.Run("local empty verdict", func(t *testing.T) {
		m, auth := newEmptyGuardManager(t, model, &emptyGuardExecutor{})
		for _, code := range []string{"empty_stream", "empty_completion"} {
			m.MarkResult(context.Background(), Result{
				AuthID:   auth.ID,
				Provider: auth.Provider,
				Model:    model,
				Error:    &Error{Code: code, Message: "upstream returned no output", Retryable: true},
			})
			assertAuthNotCoolingDown(t, m, auth.ID, model)
		}
	})

	t.Run("genuine upstream 503", func(t *testing.T) {
		m, auth := newEmptyGuardManager(t, model, &emptyGuardExecutor{})
		m.MarkResult(context.Background(), Result{
			AuthID:   auth.ID,
			Provider: auth.Provider,
			Model:    model,
			Error:    &Error{Code: "upstream_unavailable", Message: "service unavailable", HTTPStatus: http.StatusServiceUnavailable},
		})
		updated, ok := m.GetByID(auth.ID)
		if !ok || updated == nil {
			t.Fatal("expected auth to be present")
		}
		if blocked, _, _ := isAuthBlockedForModel(updated, model, time.Now()); !blocked {
			t.Fatalf("upstream 503 did not cool down the model: %#v", updated.ModelStates[model])
		}
	})
}
