package auth

import (
	"context"
	"net/http"
	"testing"
)

// An empty-completion verdict must not freeze the credential: with a single
// codex auth, a cooldown here turned every subsequent request (including
// unrelated /v1/chat/completions calls) into auth_unavailable.
func TestManager_MarkResult_EmptyCompletionDoesNotCooldown(t *testing.T) {
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	prevTransient := transientErrorCooldownSeconds.Load()
	SetTransientErrorCooldownSeconds(60)
	t.Cleanup(func() { transientErrorCooldownSeconds.Store(prevTransient) })

	const model = "gpt-6-luna"

	t.Run("model level", func(t *testing.T) {
		m := NewManager(nil, nil, nil)
		auth := &Auth{ID: "auth-empty-completion-model", Provider: "codex"}
		if _, errRegister := m.Register(context.Background(), auth); errRegister != nil {
			t.Fatalf("register auth: %v", errRegister)
		}
		result := Result{AuthID: auth.ID, Provider: auth.Provider, Model: model}
		if errMark := m.markEmptyCompletion(context.Background(), &result); !isEmptyCompletionError(errMark) {
			t.Fatalf("markEmptyCompletion error = %v, want empty_completion", errMark)
		}
		assertNoCooldown(t, m, auth.ID, model)
	})

	t.Run("auth level", func(t *testing.T) {
		m := NewManager(nil, nil, nil)
		auth := &Auth{ID: "auth-empty-completion-auth", Provider: "codex"}
		if _, errRegister := m.Register(context.Background(), auth); errRegister != nil {
			t.Fatalf("register auth: %v", errRegister)
		}
		m.MarkResult(context.Background(), Result{AuthID: auth.ID, Provider: auth.Provider, Error: errEmptyCompletion})
		assertNoCooldown(t, m, auth.ID, model)
	})

	t.Run("genuine upstream 503 still cools down", func(t *testing.T) {
		m := NewManager(nil, nil, nil)
		auth := &Auth{ID: "auth-upstream-503", Provider: "codex"}
		if _, errRegister := m.Register(context.Background(), auth); errRegister != nil {
			t.Fatalf("register auth: %v", errRegister)
		}
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
		state := updated.ModelStates[model]
		if state == nil || !state.Unavailable || state.NextRetryAfter.IsZero() {
			t.Fatalf("expected upstream 503 to cool down the model, got %#v", state)
		}
	})
}
