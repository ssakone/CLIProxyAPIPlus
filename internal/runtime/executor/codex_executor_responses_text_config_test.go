package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	_ "github.com/router-for-me/CLIProxyAPI/v7/internal/translator"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

// codexResponsesCompletedFixture mirrors the shape the chatgpt.com codex
// backend returns for gpt-6-* models: event: framing lines, a reasoning item,
// and a completed response object whose "text" field is the format
// configuration object rather than a string.
const codexResponsesCompletedFixture = "event: response.created\n" +
	`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-6-luna","output":[],"text":{"format":{"type":"text"},"verbosity":"medium"}}}` + "\n\n" +
	"event: response.output_item.added\n" +
	`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}` + "\n\n" +
	"event: response.output_item.done\n" +
	`data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"enc"}}` + "\n\n" +
	"event: response.output_text.delta\n" +
	`data: {"type":"response.output_text.delta","sequence_number":3,"output_index":1,"content_index":0,"delta":"OK"}` + "\n\n" +
	"event: response.output_item.done\n" +
	`data: {"type":"response.output_item.done","sequence_number":4,"output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"OK","annotations":[]}]}}` + "\n\n" +
	"event: response.completed\n" +
	`data: {"type":"response.completed","sequence_number":5,"response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-6-luna","output":[],"text":{"format":{"type":"text"},"verbosity":"medium"},"usage":{"input_tokens":9,"output_tokens":5,"total_tokens":14}}}` + "\n\n"

func newCodexResponsesFixtureServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, codexResponsesCompletedFixture)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestCodexExecutorExecute_ResponsesTextConfigObjectIsPreserved(t *testing.T) {
	server := newCodexResponsesFixtureServer(t)
	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"base_url": server.URL, "api_key": "test"}}

	resp, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-6-luna",
		Payload: []byte(`{"model":"gpt-6-luna","input":"Reply with exactly OK."}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse, Stream: false})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !gjson.GetBytes(resp.Payload, "text.format").IsObject() {
		t.Fatalf("expected text configuration object in payload: %s", resp.Payload)
	}
	if got := gjson.GetBytes(resp.Payload, "output.#(type==\"message\").content.0.text").String(); got != "OK" {
		t.Fatalf("message text = %q, want OK; payload=%s", got, resp.Payload)
	}
	if got := gjson.GetBytes(resp.Payload, "status").String(); got != "completed" {
		t.Fatalf("status = %q, want completed; payload=%s", got, resp.Payload)
	}
}

func TestCodexExecutorExecuteStream_ResponsesEventFramingIsPreserved(t *testing.T) {
	server := newCodexResponsesFixtureServer(t)
	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"base_url": server.URL, "api_key": "test"}}

	stream, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-6-luna",
		Payload: []byte(`{"model":"gpt-6-luna","input":"Reply with exactly OK.","stream":true}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse, Stream: true})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}

	var joined strings.Builder
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		joined.Write(chunk.Payload)
	}

	// Every event: line must stay a separate field of its own SSE frame, and
	// the completed frame must carry the text configuration object intact.
	got := joined.String()
	for _, event := range []string{"response.created", "response.output_item.added", "response.output_text.delta", "response.output_item.done", "response.completed"} {
		if !strings.Contains(got, "event: "+event+"\ndata: ") {
			t.Fatalf("event %q merged with its data line; stream=%q", event, got)
		}
	}
	var completed []byte
	for _, line := range strings.Split(got, "\n") {
		if data, ok := strings.CutPrefix(line, "data: "); ok && gjson.Get(data, "type").String() == "response.completed" {
			completed = []byte(data)
		}
	}
	if completed == nil {
		t.Fatalf("response.completed data line missing; stream=%q", got)
	}
	if !gjson.GetBytes(completed, "response.text.format").IsObject() {
		t.Fatalf("expected text configuration object in completed event: %s", completed)
	}
	if got := gjson.GetBytes(completed, "response.output.#(type==\"message\").content.0.text").String(); got != "OK" {
		t.Fatalf("completed output message text = %q, want OK; event=%s", got, completed)
	}
}
