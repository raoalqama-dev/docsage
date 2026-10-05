# docsage

Document question answering you can measure: hybrid retrieval, reranking, cited
answers and an evaluation suite, for any OpenAI-compatible LLM endpoint.

docsage answers questions over your documents with numbered citations, and ships
an evaluation suite that measures retrieval and answer quality instead of
guessing. It talks to any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio or
a hosted API) over plain HTTP with no vendor SDK, and runs its embedding and
reranking models locally.

**Status:** early development.
