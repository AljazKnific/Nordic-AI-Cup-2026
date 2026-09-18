# Backlog and handoff

Written for a session that has no memory of how any of this came about. Read
`CONTEXT.md` for the vocabulary and `docs/adr/0001` before touching a prompt.

## Where things stand

**Local score 0.745** over the 39 supplied conversations — accuracy 0.990, mean
tIoU 0.581. The floor is 0.200. `Score = 0.4 x Accuracy + 0.6 x mean tIoU`, so
evidence is the larger half and is where all remaining points are.

**On the hosted validation set the score was 0.59**, because three of nineteen
conversations exceeded the 60 s budget and a timeout costs all ten of its marks.
Both causes are now fixed (threading, then a transcription deadline) but **the
result of a validation run with both fixes in place is not yet known**. Getting
that number is the first thing to do.

Work lives on branch `aljaz-medical`, pushed to the `fork` remote
(`AljazKnific/Nordic-AI-Cup-2026`). `origin` is the team repo and is read-only
for this account. Local `main` is stale at 0.692.

## Done

| | |
| --- | --- |
| 1-3 | Pipeline, offline scoring harness, Ollama client |
| 4 | Span trimming: spans round out to whole sentences. 0.692 -> 0.745 |
| 5 | Error analysis (`tools/errors.py`) |
| 6 | **Declined with evidence** — see below |
| — | Containerised (`Dockerfile`, `docker-compose.yml`, `docs/deploy.md`) |
| — | Timeout fixes: host-wide ASR threads, whole-request budget, transcription deadline |

Ticket 6 (matching question terms against garbled transcript words) was declined
deliberately: all four wrong answers across 390 questions are garbled drug names,
worth about +0.01, but the same mechanism would let a **decoy term** match, and
hard negatives are currently perfect at 1.000 over 142 questions. Ten points of
upside against 140 of exposure. Do not rebuild it without new evidence.

## Still to do

### A: Re-run validation and read the result

**Blocked by:** None. Do this first.

Three timeouts cost ~0.12 of score; one remained after the first fix and should
now be gone. Until a clean validation run exists, nobody knows whether the
remaining gap is modelling or infrastructure — and the answer decides whether
ticket 7 is worth starting.

- [ ] A validation run completes with no conversation exceeding 60 s
- [ ] The score is recorded here

### 7: Mention selection

**Blocked by:** A

The only remaining modelling work, and larger than the backlog first estimated.
The diagnostic attributes **0.093 of tIoU** to the model naming a segment that
does not hold the annotated mention, and a further **0.091** to choosing the
wrong sentences among those the named segment does hold. 118 of 195 questions
are already the best their named segment allows, so the work is in choosing
*which mention*, not in span mechanics.

Iterate with `tools/replay.py` (frozen replies, scores in under a second).
Changing which segment the model *names* needs new replies, so that part costs a
fresh 12-minute `tools/capture_replies.py` run.

- [ ] Both variants scored over the same 39 conversations
- [ ] Ships only if it wins; the losing number recorded either way

### 8: Attempt readiness

**Blocked by:** A

Never done end to end. Would have caught the timeouts before the hosted run did.

- [ ] All 39 through the live server via `local_evaluator.py`, uncached
- [ ] Mean time per conversation recorded, and the slowest named
- [ ] Confirmation that no failure mode produces silence rather than a guess

### 9: Azure sizing, if deploying

**Blocked by:** None, but only matters if the endpoint moves off the Mac.

The container is built and verified; sizing is not. Every timing is from an M4
Pro. A CPU-only Azure VM will be slower and will reintroduce timeouts. See
`docs/deploy.md` — a GPU VM keeps the models, and everything else is an
environment variable.

### 10: Merge

**Blocked by:** A

Open a PR from `fork/aljaz-medical` to the team repo, or get write access on
`origin`. Note the fork is public.

## Numbers worth not re-deriving

Ceilings, measured with `tools/score.py --ceiling` / `--ceiling-words` (no LLM,
returns in under a second):

| Span granularity | mean tIoU |
| --- | --- |
| Whole ASR segments | 0.521 |
| Whole sentences | 0.708 |
| Runs of up to four sentences | 0.815 |
| Best word run | 0.928 |
| Best word run, calibrated offset | 0.938 |

**1.00 is not reachable.** A perfect system on today's ASR word timings caps at
about 0.963 final score. The gap is uniform quantization error, not a fixable
bias — only finer timestamps (WhisperX forced alignment) would raise it, and
that should be measured with `--ceiling-words` before any of it is built.
Accuracy is worth +0.004 in total; do not spend time there.

Timing, on an M4 Pro: ~33 s per conversation end to end, ~48 s worst case on the
longest supplied audio, against a 60 s budget that is an **average across the
whole attempt**.

## Running it

```
./.venv/bin/python -m pytest              # 34 tests, no model needed
./.venv/bin/python tools/score.py         # full score, ~12 min, buffers output
./.venv/bin/python tools/score.py --limit 5
./.venv/bin/python tools/replay.py        # frozen replies, seconds
./.venv/bin/python tools/replay.py --diagnose
caffeinate -dimsu ./.venv/bin/python api.py
```

Traps that have each cost real time:

- Use `./.venv/bin/python`, never `python3` (3.14, wrong wheels).
- Ollama is at **`127.0.0.1`**, never `localhost` — it binds IPv4 only and
  `localhost` resolves to `::1` first, failing as a bare connection refused
  while `curl` works.
- Ollama defaults to a 4096-token context and silently truncates; `num_ctx` is
  set explicitly in `ollama_client.py`.
- Transcripts for all 39 are cached in `transcripts/` (gitignored). Do not
  re-transcribe; it takes 13 minutes and the audio never changes.
- **Span faults are invisible in accuracy.** A parsing bug once halved tIoU
  while accuracy did not move at all. Judge every change by tIoU.
- Batched inference gives no speedup on CPU and collapses segments; VAD does
  nothing here. Both were measured and rejected.
- The Mac is set to sleep after 1 minute. Run the server under `caffeinate`, or
  the endpoint vanishes mid-attempt and silence ends it.
