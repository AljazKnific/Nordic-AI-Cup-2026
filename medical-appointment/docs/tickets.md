# Backlog and handoff

Written for a session that has no memory of how any of this came about. Read
`CONTEXT.md` for the vocabulary and `docs/adr/0001` before touching a prompt.

## Where things stand

**Local score 0.759** over the 39 supplied conversations — accuracy 0.990, mean
tIoU 0.605. The floor is 0.200. `Score = 0.4 x Accuracy + 0.6 x mean tIoU`, so
evidence is the larger half and is where all remaining points are.

**That number is from a full attempt through the live endpoint**, not from the
offline harness: 39 conversations, 390 questions, uncached, one POST each. No
timeouts, no failed conversations, no unanswered question. **29.5 s mean per
conversation, 45.3 s at worst — 76% of the 60 s budget** (`conversation_sample_20.mp3`,
29.8 s of it transcription). The log is `diagnostics/e2e_after.log`, which is gitignored: re-run
`local_evaluator.py` to regenerate it.

**On the hosted set the score is 0.69** (2026-09-18, no errors), up from **0.59**
before the timeout fixes. Three of nineteen conversations used to exceed the 60 s
budget and a timeout costs all ten of its marks; +0.10 is about what those three
were worth, which closes the question of whether the gap was infrastructure or
modelling. It was infrastructure.

**0.69 hosted against 0.759 local is not a regression.** The hosted conversations
are a different, unseen set — longer, on the evidence of their timings — so the
two numbers do not compare directly. The comparison that holds is 0.59 -> 0.69 on
the same hosted set.

**The margin there is thinner than it is here.** Over the 21 requests the service
sent (`sample_3` three times, which reads as a retry after its first took 51.0 s),
three conversations came within 9-11 s of the timeout: `sample_7` 50.2 s,
`sample_26` 50.9 s, `sample_31` 49.1 s, against a worst case of 45.3 s over the
supplied 39. Transcription is the pressure — 30.2 s of `sample_7` alone — and
answering ran at 1.6-2.1 s per question against 1.1-1.5 s locally. If the audio
gets longer or the host busier, that margin is where the next marks go, and
`TRANSCRIBE_BUDGET` is the lever: an environment variable, no code change. The
log is archived at `diagnostics/hosted_attempt_2026-09-18.log` (gitignored).

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
| A, 8 | Attempt readiness: all 39 through the live endpoint, clean. Numbers above |
| 7 | Mention selection: sentence choice shipped (0.745 -> 0.759); segment choice **declined with evidence** — see below |

## Declined, with the evidence

**Ticket 6 — matching question terms against garbled transcript words.** All four
wrong answers across 390 questions are garbled drug names, worth about +0.01, but
the same mechanism would let a **decoy term** match, and hard negatives are
currently perfect at 1.000 over 142 questions. Ten points of upside against 140 of
exposure. Do not rebuild it without new evidence.

**Ticket 7's second half — asking the model for the last mention.** Of the tIoU
still missing, 0.067 sits behind the model naming a segment that does not hold the
annotated passage. The pattern in those failures is that the fact is stated twice,
the patient first and the doctor in confirmation, and the annotation is usually on
the later one. A prompt that says so — *"if the consultation states the answer more
than once, name the last segment where it is stated"* — was captured over the same
39 conversations (`diagnostics/replies_last_mention.json`) and **lost**:

| prompt | accuracy | mean tIoU | score |
| --- | --- | --- | --- |
| current | 0.990 | 0.605 | **0.759** |
| last mention | 0.990 | 0.569 | 0.737 |

It made the thing it targeted worse, not better: wrong-mention loss rose from 0.067
to 0.090. The model's own choice of mention is better than the heuristic, and the
apparent "later is annotated" pattern does not survive contact with the cases where
it is not. Reproduce with `tools/replay.py --replies diagnostics/replies_last_mention.json`
if that capture is still on the machine — `diagnostics/` is gitignored, so on a
fresh clone it costs another 10-minute capture with the sentence above added to
`TASK` in `answering.py`.

An asymmetric reach — letting a passage extend further forward than back, on the
same theory — was also measured and is worth +0.002 tIoU with a bootstrap interval
straddling zero. Not shipped.

## Still to do

### 10: Merge

**Blocked by:** None.

The submission is done and the number is above. What remains is getting the work
onto the team repo.

- [x] A hosted validation run, with the endpoint served under `caffeinate` — 0.69,
      no errors, 2026-09-18
- [x] The hosted score recorded here, beside the local 0.759
- [ ] A PR from `fork/aljaz-medical` to the team repo, or write access on `origin`.
      Note the fork is public

### 9: Azure sizing, if deploying

**Blocked by:** None, but only matters if the endpoint moves off the Mac.

The container is built and verified; sizing is not, and cannot be from here. Every
timing above is an M4 Pro. A CPU-only Azure VM will be slower and will reintroduce
timeouts — at 45.3 s worst case, the headroom is 14.7 s. `docs/deploy.md` has the
levers in the order worth trying; a GPU VM keeps the models and everything else is
an environment variable.

### 11: Mention selection, if anyone returns to modelling

**Blocked by:** Nothing, but read the declined note above first.

What is left after the sentence work, attributed by `tools/replay.py --diagnose`:

| cause | n | recoverable tIoU |
| --- | --- | --- |
| wrong-sentences | 49 | 0.114 |
| wrong-mention | 33 | 0.067 |
| answered-no | 4 | 0.019 |
| best-available | 109 | 0.000 |

Two of those are dead ends already paid for: `answered-no` is ticket 6, and the
prompt route into `wrong-mention` is the table above. `wrong-sentences` is the one
with headroom left, and it is honest headroom — the passage was reachable from the
segment the model named and we chose the wrong sentences inside it. Iterate with
`tools/replay.py` (frozen replies, under a second); only a change to which segment
the model *names* needs a fresh 10-minute capture.

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

Timing, on an M4 Pro, measured over a whole 39-conversation attempt through the
live endpoint: **29.5 s mean, 45.3 s worst** against a 60 s budget that is an
**average across the whole attempt**. Transcription is the variable half — 29.8 s
of that worst case — and answering is steady at 1.1-1.5 s per question once the
shared prefix is warm.

## Running it

```
./.venv/bin/python -m pytest              # 36 tests, no model needed
./.venv/bin/python tools/score.py         # full score, ~12 min, buffers output
./.venv/bin/python tools/score.py --limit 5
./.venv/bin/python tools/replay.py        # frozen replies, seconds
./.venv/bin/python tools/replay.py --diagnose
./.venv/bin/python local_evaluator.py     # a whole attempt through the live server, ~20 min
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
- Python caches bytecode on mtime **and size**, so an edit that changes neither
  (flipping a constant back after a experiment, say) runs the stale `.pyc` and
  quietly reports the old numbers. Clear `__pycache__` after any such edit.
- The Mac is set to sleep after 1 minute. Run the server under `caffeinate`, or
  the endpoint vanishes mid-attempt and silence ends it.
