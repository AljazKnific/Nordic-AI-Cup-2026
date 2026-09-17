# Backlog

Tickets 1-3 are done (commit 94408e8). What follows is the remaining work, in
dependency order. Blocking edges are real: a ticket whose blockers are unfinished
cannot be measured, only guessed at.

Baseline to beat, measured over all 39 supplied conversations:

| | mean tIoU | score |
| --- | --- | --- |
| Pipeline before ticket 4 | 0.493 | 0.692 |
| Segment bounds only (ceiling) | 0.521 | 0.713 |
| **Pipeline after ticket 4** | **0.581** | **0.745** |
| Whole sentences (ceiling, runs of up to 4) | 0.815 | 0.889 |
| Word-level trimming (ceiling) | 0.928 | 0.957 |

Accuracy is unchanged at 0.990 throughout; every point moved is evidence.

Accuracy is 0.990 and is not the problem. Every remaining point of consequence
is in the evidence half.

---

## 4: Make quote trimming actually beat segment bounds

**Blocked by:** None (can start immediately)

**What to build:** Evidence spans that are worth more than pointing at whole ASR
segments. The mechanism exists — the answering model names a segment and quotes
what it read, and `resolve_span` matches that quote against word timings inside
that segment — but it currently scores 0.493 against a 0.521 segment-bounds
ceiling, so it is not yet earning its place. The word-level ceiling is 0.928,
so the headroom is roughly +0.26 of final score.

Diagnose before changing anything: for each annotated yes question, compare the
predicted span against the gold span and against what the best word run in the
named segment would have scored. That separates three different failures —
the model naming the wrong segment, the quote failing to match, and the quote
matching but covering the wrong words.

**Acceptance criteria:**
- [x] A diagnostic exists that attributes each lost point to one of those three causes
- [x] Mean tIoU over all 39 conversations beats 0.521
- [x] The quote is still matched only inside the named segment
- [x] Prompts still open with a byte-identical transcript header (ADR-0001)
- [x] Tests at the `answer_conversation` seam cover any new behaviour

**Done.** `tools/capture_replies.py` records one real model reply per question
once; `tools/replay.py` then scores any span change against them in a second
instead of twelve minutes, and `--diagnose` does the attribution.

What it found: none of the three suspected causes was the main one. The quote
fails to match its segment **once in 390 questions**. The model names a usable
segment almost always. What was actually wrong is that the unit was wrong --
the ASR divides speech into ~3 s segments that cut sentences in half, and 68 of
the 195 annotated passages cross a segment boundary, so a span trimmed to the
quote inside one segment is routinely half a sentence. Pointing at whole
segments is worth 0.521; whole sentences are worth 0.708 and contiguous runs of
them 0.815.

So the span is now rounded out to the **sentences** the quote lies in, taken
from the punctuation the ASR already puts on its word tokens. The quote is
still matched only inside the named segment; only the widening crosses the
boundary, which cannot pull the span to a different mention.

## 5: Error analysis pass

**Blocked by:** None (can start immediately)

**What to build:** A readout of what is actually wrong, rather than a single
score. The pipeline already logs every fallback with a `FALLBACK` prefix; this
turns those and the per-question results into counts.

**Acceptance criteria:**
- [x] Accuracy broken down by question type, with the specific questions that failed
- [x] A count of how often each fallback layer fires across the 39 conversations
- [x] A count of how often a dose, number or drug name is transcribed wrong
- [x] A written conclusion on whether ticket 6 is worth building

**Done**, as `tools/errors.py`. Findings over all 39 conversations:

| question type | correct | n |
| --- | --- | --- |
| positive | 0.979 | 195 |
| hard_negative | 1.000 | 142 |
| off_topic | 1.000 | 53 |

Every wrong answer is a positive answered *no*, and there are four of them:
`sample_5_yes_q04` (Activelle), `sample_19_yes_q02` (Esomeprazole),
`sample_19_yes_q03` (Airomir), `sample_82_yes_q01` (molluscs).

Fallback layers over the same 390 questions: `quote-unmatched` fires **once**,
and no other layer fires at all. The quote path is not where anything is being
lost. The deadline and answerer-exception layers cannot fire in a replay and
did not fire in the live run either.

Entities: of 31 drug names and numbers named by positive questions, 10 are not
spelled the same in the transcript and 4 of those have a near-match, which is
ASR garbling -- Esomeprazole heard as "Isameprosol", Airomir as "Aromere",
Ibumetin as "Ibumet", Panodil as "Panadil".

## 6: Question-vocabulary entity matching

**Blocked by:** 5

**What to build:** Match entities named in a conversation's own ten questions
against garbled transcript text, so an unseen drug name spelled correctly in the
question can be recognised in an imperfect transcript.

Use the question terms **only after ASR**, never as hotwords or `initial_prompt`.
Hard-negative questions carry decoy drug names by construction — Pantoprazole
where Esomeprazole was spoken — so priming transcription with them risks
corrupting the transcript the whole pipeline rests on. A bad match costs one
question; a poisoned transcript costs the conversation.

Do not build this unless ticket 5 shows entity errors are actually happening.

**Decision after ticket 5: do not build.** The entity errors are real but they
are rare and cheap, and the fix is dangerous in exactly the place the pipeline
is currently perfect.

Upside: three of the four wrong answers are a garbled drug name. Recovering all
three is +0.008 accuracy and, because a positive answered *no* also scores a
tIoU of 0, perhaps +0.011 mean tIoU -- about **+0.01 of final score**.

Downside: the mechanism is fuzzy-matching a question's term against garbled
transcript words, and a hard negative's decoy term is a near-miss on a real one
by construction. The same matcher that maps Esomeprazole onto "Isameprosol" can
map the decoy Pantoprazole onto it too, turning a correct *no* into a confident
*yes*. That risk is spread over 142 hard negatives currently answered at 1.000.
Ten points of upside against a hundred and forty of exposure is the wrong bet.

Revisit only if hard-negative accuracy is ever measured with slack in it.

**Acceptance criteria:**
- [ ] No lexicon derived from the 39 training conversations (evaluation drugs are unseen)
- [ ] Terms used only after transcription
- [ ] Score over all 39 conversations does not regress

## 7: A/B meaning-based mention selection

**Blocked by:** 4

**What to build:** When a fact is stated more than once, choose the mention by
what the question means rather than taking the first match. Ceiling is about
+0.046: only 15 of 195 annotated yes questions have a duplicate mention at all.

Ship only if the A/B beats plain quoting. It is a fix for a problem that may not
exist — check first whether the model already picks the annotated mention.

**Acceptance criteria:**
- [ ] Both variants scored over the same 39 conversations
- [ ] Ships only if it wins, and the losing variant's number is recorded either way

## 8: Attempt readiness

**Blocked by:** 4

**What to build:** Evidence that the real path holds under the real rules.
Budget is 60 s per conversation **averaged over the whole attempt**, not per
request — being slow early takes marks off conversations that are then never
sent. Five consecutive timeouts end an attempt outright, so silence is the worst
possible failure.

**Acceptance criteria:**
- [ ] All 39 conversations run through the live `api.py` server via `local_evaluator.py`, uncached
- [ ] Mean time per conversation recorded and under 60 s, with the slowest named
- [ ] Confirmation that no failure mode produces silence rather than a guess
- [ ] Model weights confirmed present locally, so the attempt needs no network
