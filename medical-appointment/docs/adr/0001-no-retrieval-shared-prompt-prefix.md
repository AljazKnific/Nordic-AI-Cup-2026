# No retrieval; one shared prompt prefix per conversation

A conversation is short enough to fit in the answering model's context whole
(~10,600 tokens), and nothing is reused between requests, so a per-request vector
index buys no recall we don't already have. We therefore put the **entire
transcript** in every prompt, as a **byte-identical prefix**, with the question
appended after it — which lets Ollama reuse the cached attention state across all
ten questions.

This is measured, not assumed. Prompt processing dominates: qwen3:14b reads ~10,600
tokens to write ~146, at ~188 tok/s prompt against ~26 tok/s generation, so ~91% of
answering time was re-reading the transcript once per question. Sharing the prefix
took ten questions from **86.7 s to 15.7 s**; calls two through ten fell from 8.5 s
to 0.7 s each. Against a 60 s per-conversation budget, that difference is the whole
attempt.

## Considered Options

- **Per-question retrieval (ChromaDB + a local embedding model)**: rejected. It gives
  every question a *different* prompt prefix, so the cache never hits and we pay the
  86.7 s path. It also narrows the context, which is actively dangerous here: hard
  negatives are near-misses ("0.15 mg" against a transcript saying "0.3 mg"), so a
  retrieved chunk that omits the contradicting mention turns a correct *no* into a
  confident wrong *yes*.
- **Batching all ten questions into one call**: rejected. Cheapest of all, but it
  loses answer isolation — the model's reading of one question contaminates the next,
  and hard negatives are exactly where that shows.
- **Running the ten calls in parallel**: rejected. Concurrent requests evict each
  other's shared prefix, giving back the saving they were meant to exploit.

## Consequences

The prefix is load-bearing and fragile in a way that is invisible at a glance:

- Any prompt edit that puts the question, or anything else per-question, **before**
  the transcript silently costs ~70 seconds per conversation. Task text goes *after*
  the transcript header, always.
- The ten calls must stay **sequential**.
- Anything that varies the transcript text per question — retrieval, trimming,
  re-ordering, per-question highlighting — breaks the prefix. Evidence localisation
  therefore works from segment indices *within* the shared transcript rather than by
  rewriting it.
