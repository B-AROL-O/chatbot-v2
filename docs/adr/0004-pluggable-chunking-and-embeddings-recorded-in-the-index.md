# 0004. Make chunking and embedding pluggable, with each index recording how it was built

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Retrieval for the `documents` domain is hard-coded:

- **Chunking**: fixed 1000-character windows with 200 characters of overlap
  (`ingestion/ingest_docsvc.py`, `chunk_text`).
- **Embeddings**: the local `paraphrase-multilingual-MiniLM-L12-v2` model, named
  separately in `ingestion/ingest_docsvc.py` and `tools/docsvc_rag.py`.

That model reads at most 128 tokens. A 1000-character Italian chunk is about
208 tokens (measured 2026-11-19), so each chunk is embedded from roughly its first
615 characters. With a new chunk every 800 characters, about 23% of the text is never
embedded in any chunk; only the BM25 index sees it. Fixed windows also cut tables
in half.

Better chunkers and models are available, but switching everyone at once has three
problems:

- Deployments of this project may want to keep the retrieval they have, or switch
  back.
- `sme-chatbot-eval-retrieval` can only compare configurations if their indexes can
  exist side by side.
- `read_pages` rebuilds a page from its chunks using the fixed chunker's overlap
  (`tools/docsvc_rag.py`, `_join_chunks` imports `CHUNK_OVERLAP`), so any other
  chunker breaks it.

## Considered options

- **Replace in place**: change the default chunker and model for everyone. Simplest,
  but every deployment must reindex at once, there is no way back, and no A/B
  comparison.
- **Configurable through environment variables only**: easy to add, but nothing ties
  an index to the settings it was built with. Changing the model setting against an
  existing index silently returns wrong neighbours, or fails on a vector-dimension
  mismatch.
- **Pluggable, with the index recording its configuration** (chosen).

## Decision

We will make chunking and embedding pluggable behind two small interfaces:

- **Chunker**: takes a document's pages and returns chunks. Each chunk carries its
  start and end page and its character offsets within the page, and optionally its
  heading path.
- **Embedding provider**: separate `embed_documents` and `embed_query` (some models
  need different prefixes for queries and passages), plus `dimension` and
  `max_input_tokens`.

Built-in implementations, selected by configuration:

| Kind | Implementation | Behaviour |
| --- | --- | --- |
| Chunker | `fixed` | **Default**: today's 1000/200 windows |
| Chunker | `recursive` | Splits at headings, then paragraphs, then sentences |
| Chunker | `markdown` | One section per heading; tables kept whole; heading path prepended as context |
| Embedding provider | `sentence-transformers` | **Default**: today's model, run locally |
| Embedding provider | `openai` | Any `/v1/embeddings` endpoint: OpenAI, Azure, vLLM, TEI, Ollama, a LiteLLM proxy |

There is no plugin mechanism: a new implementation is added to the built-in registry.

Ingestion writes the configuration into the collection's metadata:

- provider, model and vector dimension;
- chunker and its parameters;
- a format version.

Several configurations can coexist as separate collections. The query side builds
its embedder from that record, and refuses to start when the configured settings
contradict it. An index without a record is treated as the default configuration.

`read_pages` will rebuild pages from the stored character offsets, not from the
chunker's parameters. This decoupling lands first.

The defaults must reproduce today's chunk ids and texts exactly, pinned by a test.
For the non-fixed chunkers, chunk sizes are measured in the embedding model's tokens.
Ingestion warns when a chunk exceeds `max_input_tokens`.

## Consequences

- A deployment that changes nothing keeps today's retrieval. Switching means
  ingesting into a new collection and changing configuration. Rolling back is a
  configuration change.
- Retrieval changes can be compared with `sme-chatbot-eval-retrieval` on indexes
  built side by side.
- More configuration surface and a larger test matrix. The BM25 index
  (`lexical_fts.sqlite3`) is built per collection, on the same chunk ids.
- A remote embedding provider receives every chunk at ingestion and every query at
  search time, and adds a network round trip to each search. Using one is a
  data-governance decision for each deployment; the docs must say so.
- Out of scope: the legacy `sample_cdo` and `sample_manuals` pipelines
  (`ingest_sample_*.py`).
- Revisit if one configuration clearly wins in the evals and no deployment uses the
  others: the unused implementations can then be dropped.
