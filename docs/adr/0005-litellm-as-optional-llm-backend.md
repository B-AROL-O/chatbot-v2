# 0005. Add LiteLLM as an optional in-process LLM backend, behind the existing `llm_api` seam

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The backend talks to the LLM through the OpenAI SDK at `LLM_BASE_URL`, so it
already works with any OpenAI-compatible server (vLLM, Ollama, Azure OpenAI,
OpenRouter, ...). `LLM_API` chooses between `chat_completions` and `responses`.
`core/llm_api.py` adapts either one into chat-completions-shaped chunks, so the
streaming loop in `core/chat.py` is the same on both.

Two things remain hard:

- **Other providers.** Providers without an OpenAI-compatible endpoint, or whose
  native API offers more (Anthropic, Gemini, Bedrock, ...), need a gateway in front.
- **Late setup errors.** A missing key, or a setting the model rejects, only shows up
  at the first chat request.

Langfuse tracing works by swapping in Langfuse's OpenAI wrapper (`api/main.py`,
`_get_client`).

The same need was solved in [citefield](https://github.com/aleche28/citefield) with
a backend protocol, a LiteLLM adapter, and a provider-setup check before the first
call (its ADR-0005 and ADR-0023).

## Considered options

- **Keep the OpenAI client only, and document a LiteLLM proxy** (or another gateway)
  as a separate service. No code change, but one more service to run, and setup
  errors still surface at the first request.
- **Replace the OpenAI client with the LiteLLM SDK.** A single code path, but every
  deployment carries a heavy dependency, and setups that work today change behaviour.
- **LiteLLM as an optional backend behind `llm_api.py`** (chosen).

## Decision

We will add `litellm` as a third value of `LLM_API`:

- `open_stream` calls `litellm.completion(..., stream=True)`, whose chunks already
  have the chat-completions shape. `stream_chat` does not change.
- `LLM_MODEL` then takes LiteLLM's `provider/model` form.
- The OpenAI client and `chat_completions` stay the default.
- LiteLLM is an optional dependency extra. Selecting `litellm` without it installed
  is a startup error.

When `LLM_API=litellm`:

- **Startup check.** The backend checks at startup that the provider's environment
  variables are set, and refuses to start, naming the missing variable.
- **No OpenAI client.** `api/main.py` does not build the OpenAI client (or its
  Langfuse wrapper), and `open_stream` gets `client=None`. Each request then has
  one LLM client and one Langfuse integration, not two.
- **Tracing.** Langfuse tracing goes through LiteLLM's Langfuse callback instead of
  the OpenAI wrapper.

There is no plugin mechanism. Running a LiteLLM proxy and pointing `LLM_BASE_URL` at
it remains a supported, zero-code alternative, and is documented as such.

## Consequences

- Native providers work without a gateway, and setup mistakes fail at boot rather
  than on a user's question.
- `core/chat.py`'s tool-call accumulation is untouched. Providers may still differ in
  how they stream tool-call fragments and in their error texts:
  `_is_tools_with_reasoning_rejection` matches error text. Both need tests on
  recorded streams.
- A provider counts as supported only after an `sme-chatbot-eval` run on it.
- Open follow-up: whether the published Docker image includes the extra, or a
  separate image variant does.
- Revisit if provider differences force special cases into the streaming loop. At
  that point a per-provider backend interface would be the cleaner seam.
