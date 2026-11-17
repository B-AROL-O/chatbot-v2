# Changelog

Release notes for `chatbotv2/v*` releases. Each release is also listed on the
[GitHub Releases page](https://github.com/B-AROL-O/chatbot-v2/releases).

## Unreleased

First public release. Earlier versions were developed in a private repository,
and their history is not carried over.

- FastAPI backend with one streaming chat endpoint (`POST /chat/message`, SSE),
  and a thin Gradio client over HTTP
- Three data domains: CDO records (read-only SQL + semantic search), machine
  manuals, and a DocumentService library with hybrid dense + BM25 retrieval and
  query-time access filtering
- Automated eval runner (`sme-chatbot-eval`) and retrieval benchmark
  (`sme-chatbot-eval-retrieval`) for golden question sets
- Docker images for `linux/amd64` and `linux/arm64`, with optional log shipping
  to Loki through Grafana Alloy
