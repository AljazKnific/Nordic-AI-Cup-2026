# Backlog and handoff

Written for a session that has no memory of how any of this came about. Read
`CONTEXT.md` for the vocabulary and `docs/adr/0001` before touching a prompt.

## Where things stand

**Local score 0.768** over the 39 supplied conversations — accuracy 0.990, mean
tIoU 0.621. The floor is 0.200. `Score = 0.4 x Accuracy + 0.6 x mean tIoU`, so
evidence is the larger half and is where all remaining points are.

**Read that number carefully: 0.767 is from replayed replies, not a live run.**
It is 0.759 re-scored with the fitted extent constants (ticket 12), over the
frozen replies in `diagnostics/replies.json`. The span code is a pure function of
the reply, so replay is exact for a change like this one — but it is not an
attempt, and nothing about timing or the endpoint is re-verified by it.

**The last number from a full attempt through the live endpoint is 0.759**: 39
conversations, 390 questions, uncached, one POST each. No timeouts, no failed
conversations, no unanswered question. **29.5 s mean per conversation, 45.3 s at
worst — 76% of the 60 s budget** (`conversation_sample_20.mp3`, 29.8 s of it
transcription). The log is `diagnostics/e2e_after.log`, which is gitignored:
re-run `local_evaluator.py` to regenerate it. **Re-run it before the next
submission** — the answerer's signature changed with the per-call timeout
(ticket 13), and only a live attempt exercises that path.

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
| 12 | Extent constants fitted under leave-one-out CV. 0.759 -> 0.767 |
| 13 | Abbreviation guard in `sentences_of`; per-call LLM timeout bounded by the budget |
| 14 | Second pass: **declined with evidence** — loses in both wordings. See below |

## Where the remaining loss actually is (2026-09-19)

Measured with `tools/disagreements.py`. Of 0.382 of tIoU lost:

| | n | loss | share |
| --- | --- | --- | --- |
| gold fully covered, we returned extra | 104 | 0.152 | 40% |
| partly missed *and* extra returned | 68 | 0.181 | 48% |
| returned too little / misplaced | 19 | 0.028 | 7% |
| answered no | 4 | 0.021 | 5% |

**Returning more than the passage is involved in 88% of the loss.** But it is
mostly not fixable by returning fewer sentences: of the 104 clean
over-inclusions, **82% already return exactly the sentences gold occupies** and
overshoot by a median 0.38 s *inside* them. Only 18% take two sentences where
one would do.

Two structural facts that close off several plausible-sounding ideas:

**Gold passages are single sentences, not exchanges.** 72% span exactly one
sentence, 19% two. Only 2% of the one-sentence ones open with a question. The
comment in `answering.py` that justified `REACH_SECONDS` as buying back
"doctor's question through patient's answer" exchanges was **wrong**, and is
corrected. The parameter survived fitting; the theory did not.

**So speaker diarisation would make this worse, not better.** A speaker turn is
coarser than a sentence, and gold is one sentence 72% of the time. Tested the
cheap proxy too — "is the previous sentence a question?" as a stand-in for a
speaker change, on the 1-vs-2 decision — and it splits 24% against 32%, which is
not a signal.

**Transcription is not the constraint either.** The word timings we already have
support a ceiling of **0.929** mean tIoU (`tools/score.py --ceiling-words`)
against the 0.621 we return. Better ASR, forced alignment and diarisation all
raise a ceiling that is already 0.31 above where we stand — and forced alignment
has since been measured and does not even do that (2026-09-20, below). The gap is in
*choosing* sub-sentence bounds from timings we hold, not in getting better ones.

## Is "extract less" a lever? Capped at +0.016, and every gate loses (2026-09-20)

The standing intuition is that we hand back too much around the annotated
passage and that a prompt could tell the model to be terser. Measured with
`tools/extent.py`, which reproduces all of the below in a second from the frozen
replies.

**The direction is right.** Over the 191 questions answered yes, dropping every
second we return that gold does not would be worth **+0.173 mean tIoU**; covering
every second of gold we miss would be worth **+0.072**. Over-inclusion is the
larger half, as `tools/disagreements.py` already said.

**But almost none of it is reachable by returning fewer sentences.** We return
more sentences than the annotation touches on 25 of 191 questions and *fewer* on
31; the modal case is 113 questions where we and gold occupy the same single
sentence and we score 0.714 on it. The 25 over-runs are the visible failure —
they score 0.372 — but they are 13% of the set, and the blunt fix takes the 22
questions where two sentences are right (0.836) with them:

| how far a passage may run | mean tIoU | delta |
| --- | --- | --- |
| always one sentence | 0.5918 | **-0.0419** |
| as many as the model's quote spans, capped at two | 0.6175 | **-0.0163** |
| today: up to two | 0.6338 | — |
| always three | 0.6300 | -0.0037 |
| **oracle**, as many as gold touches | 0.6597 | **+0.0260** |

**That oracle row is the finding.** A signal that decided "one sentence or two"
*perfectly* — a new prompt field, a second pass, a classifier, anything — is
worth **+0.026 mean tIoU, or +0.016 of final score**, and only if it is never
wrong. The one time the model has been asked to make a comparable judgement it
came out below the heuristic (ticket 14). This is the ceiling on the whole
"tell it to extract less" direction, and it is smaller than the +0.0131 the
extent fitting already collected for free.

**The quote's own width is not the signal either.** When the model's quote lands
in one sentence, gold is one sentence 78% of the time; when it spans two or more,
62%. A gate needs those to differ and they barely do — hence the -0.0163 above.

**And it is not really a width problem.** Of the questions where the ranking's
shortlist held a better candidate than the one it chose, 22 wanted a *shorter*
span and 14 wanted a *longer* one. A rule that only ever narrows cannot collect a
third of the available gain. The full oracle over that shortlist is +0.072 mean
tIoU, which is the same number ticket 14 was built against and lost.

**Three more narrowing ideas, each measured and each negative.** Penalising a run
whose extra sentence adds no question word the anchor sentence did not already
have: **-0.0127 to -0.0157** at every penalty from 0.25 to 5.0. `LENGTH_PENALTY`
away from its fitted 0.2 in either direction: -0.019 at 0.1, -0.019 at 0.3.
`ANCHOR_WEIGHT` away from 0.5: -0.026 at 0.25, -0.019 at 1.0. The ranking sits in
a sharp local optimum in every direction tried.

**A sentence-numbered transcript header was considered and is not worth a
capture.** Numbering sentences instead of ASR segments would let the model name
the unit we actually return, and it is 1.12x the header size — affordable. But
the proxy kills it: returning *just* the sentence the model's quote lands in
scores **0.559 against today's 0.634**, so the reach-and-run machinery it would
replace is worth +0.075 on its own, and the quote already picks the best single
sentence 76% of the time against a perfect-naming ceiling of 0.706. The model
would have to name sentences near-perfectly to break even on a design that starts
0.075 behind.

**Nor is trimming the header.** The per-segment timestamps the model never uses
are 20% of it, ~112 tokens a conversation, ~0.6 s of prefill once per
conversation against a 60 s budget whose variable half is 30 s of transcription.
Not a timing lever.

**What this leaves.** Extent is fitted out and the prompt cannot reach past
+0.016 even perfectly. The remaining evidence loss is *which* sentence, not how
far it runs, and that has now been attacked three times — last-mention (0.737),
the gated second pass (0.749/0.752), and the gates above. The untried lever is
the one the ceiling table points at: our 0.634 against 0.929 for the best word
run the existing ASR timings already support. Sub-sentence bounds, not fewer
sentences.

## WhisperX forced alignment: measured, and the ceiling goes **down** (2026-09-20)

The standing note said the word-run ceiling of 0.929 could only be raised by
finer timestamps, and named WhisperX forced alignment as the way to get them.
Built (`tools/align.py`, in its own `.venv-align`) and measured
(`tools/timings.py`). **It is worse, in every form the question can be asked.**

| | mean tIoU |
| --- | --- |
| best word run, faster-whisper timings | **0.929** |
| best word run, WhisperX forced alignment | 0.897 |
| faster-whisper, calibrated offset | **0.942** |
| WhisperX, calibrated offset | 0.936 |
| an oracle taking each edge from whichever source is better | 0.9415 |

And on the real system, replaying the frozen replies through aligned
timings — the segment text and indices are untouched, so the same captured
replies are still valid:

| | accuracy | mean tIoU | score |
| --- | --- | --- | --- |
| today | 0.990 | 0.621 | **0.768** |
| aligned timings | 0.990 | 0.585 | 0.747 |
| aligned, with `PAD_SECONDS` re-fitted to them (0.05) | 0.990 | 0.592 | 0.751 |

**Why it loses is the useful part.** Forced alignment does exactly what it
promises: its word bounds hug the phonemes. Against our timings a word starts
0.08 s later (median) and runs 0.16 s instead of 0.24 s. But an annotated
passage is *not* drawn at phoneme onsets — it sits out in the silence around the
speech, which is where faster-whisper's looser bounds already are. Measured as
the distance from an annotated edge to the nearest boundary each source offers:

| | start | end |
| --- | --- | --- |
| faster-whisper | 0.08 s | 0.06 s |
| WhisperX | 0.10 s | 0.08 s |

Sharper acoustics, further from where a human drew the line. The residual 0.07
of ceiling is not quantisation we can dissolve with better timings; it is the
annotation's own margin, and a constant offset already recovers most of what
there is (0.929 -> 0.942).

**Cost was never the obstacle**, for the record: alignment runs at 68x realtime,
1.8 s per conversation on this machine, and placed all but 1 of 12,316 words. If
it had helped it would have been affordable.

**What this closes.** Every remaining item on the evidence side is about
*choosing* the span, not measuring it. The ASR timings we already hold express
the annotation better than a purpose-built aligner does, and the gap from 0.621
to 0.929 is entirely in the choosing.

Two things were left behind that are worth keeping:

- `TRANSCRIPTS_DIR` now points any dev tool at an alternative transcript cache,
  so a timing experiment is scored through the same code as everything else:
  `TRANSCRIPTS_DIR=transcripts_aligned ./.venv/bin/python tools/replay.py`.
- `tools/timings.py` compares any two caches head to head — raw ceiling,
  calibrated ceiling, where the boundaries moved, and how far the annotation
  sits from each. That is the measurement any future ASR change should have to
  clear, and it takes about a minute.

**The trap, if anyone re-runs the alignment.** `whisperx.align` re-splits the
transcript into one segment per sentence and returns *more* segments than it was
given (52 for 26 on the first conversation). Zipping the result against the
input assigns each segment an earlier one's timings, which scores 0.485 and
looks entirely plausible in the file. Read the flat `word_segments` list
instead — and do not cut it by count either, because the two tokenisers disagree
about hyphens ("anti" + "-inflammatory" against "anti-inflammatory") and one
such word shifts every timing after it. `tools/align.py` reconciles the two
streams on their letters; both failures are written into its docstring.

## The 0.61 run, and what it taught (2026-09-19)

A hosted attempt after the fitting work scored **0.61**, down from 0.69, with two
conversations reported as `timed out after 60 seconds. All 10 questions were
scored wrong.` Two lost conversations is 20 of ~200 marks, which is the whole
drop — the modelling changes were not the cause.

**The server did not fail either request.** It logged 19 POSTs and 19 `200 OK`,
worst case 50.1 s. Of the two the evaluator failed:

- `conversation_sample_3.mp3` was answered in **35.0 s by our clock** and still
  timed out at the evaluator's 60 s. **25 s was spent somewhere this process
  could not see.**
- `conversation_sample_46.mp3` never appears in the log at all. Its body was
  never read.

**The cause: we were budgeting the wrong 50 seconds.** `predict` started its
clock after FastAPI had already received and parsed the request body. A
conversation is a 3-4 MB MP3, about 5 MB base64'd, and uploading it is a real
part of the evaluator's 60 s. It was not part of ours.

Fixed by stamping arrival in an HTTP middleware (`api.py`), before the body is
read, and budgeting from that. `TIMING` now opens with `arrival-to-work`, and
every request logs `REQUEST served in N s wall`. On loopback both read 0.0 s and
identical totals; **on the hosted path that field is the measurement that was
missing**, so read it first after the next attempt.

Two smaller faults found in the same log:

- **`MIN_TIMEOUT_SECONDS` clamped the remaining budget upward**, so with under a
  second left we sent a doomed 1 s request to Ollama that failed *and* spent what
  remained. It now refuses below 3 s and guesses instead. Cost: one question.
- **`logging.basicConfig` ran after `from example import predict`**, which loads
  and exercises both models, so every warm-up timing was silently discarded.
  Moved above the imports; timestamps added.

And one design fault that had been costing marks since before this run: a fixed
`TRANSCRIBE_BUDGET=34` against `REQUEST_BUDGET=50` was never consistent with ten
questions at 2.0-2.5 s each (34 + 25 = 59 > 50), which is why the earlier hosted
run guessed its last few questions on every long conversation. The transcription
deadline is now derived from what the remaining questions will cost, with a floor
on *work seconds* so a slow upload can never leave transcription with nothing.

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

**Within-sentence trimming — stripping "So," / "Okay," off a passage edge.**
**Built and measured on 2026-09-19: -0.0056 mean tIoU, interval entirely below
zero, and all 39 folds chose it off.** The switch is `TRIM_OPENERS` and it is
still on the fit grid, so this can be re-checked in seconds rather than rebuilt.

The reasoning was sound and the annotation simply does not agree with it: a gold
passage starts a median 0.14 s after the sentence carrying it, about what "So,"
costs, but the annotators evidently keep those openers more often than they drop
them. The earlier aggregate argument for declining it also holds. Gold passages
are already close to sentence-aligned — 68.7% of gold starts fall within
0.25 s of a sentence start, 81% of ends within 0.25 s of a sentence end — and the
sentence run covering a gold passage exceeds it by a **median of 0.20 s**. A
leading discourse marker is 0.3-0.5 s, so trimming one cuts into gold more often
than it tightens onto it. Worse, in **15.9%** of cases gold is *longer* than the
sentences we point at: we under-reach there, and trimming widens the miss.

**A constant pad that *widens* the finished passage.** Built as a sixth fitted
parameter to buy back the 15.9% of gold passages that run longer than our
sentences. Fitting chose 0.0 in all 39 folds — but only because the grid ran
from zero upwards. **Widening was on the grid and shrinking was not**, which is
a lesson about search design rather than about padding: a one-sided grid can
only ever return the boundary.

With negatives added, all 39 folds chose **-0.05**, an inset, worth **+0.0023
mean tIoU, CI +0.0004 to +0.0044**. Shipped. It is a small number and that is
the finding: an inset large enough to matter cuts into the passage. -0.10 is
already worse than nothing and -0.30 costs 0.06. **The overshoot is not a margin
to be shaved.**

**Structured JSON output (Ollama `format`) and regex parser fallbacks.** Both
target malformed replies. Across every log ever captured — `diagnostics/*.log`,
including two full 390-question attempts — there are **2** unreadable replies and
**4** unmatched quotes. There is no problem here to fix, and constrained decoding
would change the model's output distribution for no measured gain.

**The abbreviation guard was shipped, but it is unvalidated, not validated.**
`sentences_of` no longer breaks a sentence after `dr. mr. mrs. ms. prof. st.`,
which removes 39 of the 2,132 boundaries the ASR emits over the 39 conversations.
It changed the score by **nothing** (0.618 either way) and the whole-sentence
ceiling by **nothing** (0.7078 either way, and 0.8154 for runs of four). It was
kept because it is a correctness fix that costs six tokens and creates no
run-ons — the longest sentence is 8.6 s with the guard and without it. Do not
cite it as a source of points; it is not one.

Titles **only**, and deliberately. The other abbreviation-shaped tokens here are
`no.` (14 occurrences) and `am.` (4), and in a consultation both are genuine
sentence ends — "No." answers a question, "I am." closes one. Guarding those
would manufacture run-ons, which cost far more than the split they prevent.
Unit abbreviations (`mg.`, `ml.`) appear **zero** times; adding them would be
speculation dressed as a fix.

**Reaching less far forward than back.** `tools/disagreements.py` shows the
error is lopsided by *count*: our span ends late 113 times against 35 early, and
runs long 118 times against 48 short. That looks like a reach that is too
generous forward — and it is the opposite of the asymmetric reach rejected
earlier, which reached *further* forward, so it was worth its own test.

It lost: **-0.0083 under nested CV, CI -0.0215 to +0.0000**, and 37 of 39 folds
chose the two halves *equal* anyway. Two lessons, both worth keeping:

- **Count lopsidedness is not effect size.** The median end error is **+0.08 s**.
  Most of those 113 late endings are late by nothing at all; the cost is in the
  p90 of +2.10 s, a tail that a constant reach cannot address.
- **An extra parameter has a price.** Splitting one constant into two made the
  nested-CV result *worse*, not merely flat. Six parameters already sit close to
  what 195 passages can support.

Reverted rather than kept at zero, unlike `PAD_SECONDS`: `REACH_BACK = REACH`
binds once at import, so a fitter setting `REACH_SECONDS` would silently leave
the halves stale. A dead parameter that can quietly report the wrong number is
worse than no parameter.

**Re-wording what the first pass quotes.** The quote instruction is at a local
optimum, measured in four directions against the frozen replies (baseline
0.768, `tools/replay.py`, seconds per run):

| change to the quote | score |
| --- | --- |
| baseline | **0.768** |
| trimmed to its first sentence | 0.734 |
| trimmed to its last sentence | 0.746 |
| trimmed to the sentence sharing most question words | 0.754 |
| widened out to the whole sentences it touches | 0.759-0.765 |

**36% of quotes carry more than one sentence while 72% of gold passages are a
single one**, so "quote exactly one sentence" looks like free points. It is not:
the extra words are load-bearing, because they widen the anchor the candidate
ranking scores `kept` against. Trimming and widening both lose, which is what a
local optimum looks like. The trim-to-best-word variant is the closest offline
proxy for actually asking the model, and it loses by 0.014.

**Two smaller first-pass ideas, both dead on the measurement:**

- *Reordering the reply JSON so the quote precedes the segment index*, on the
  theory that the model commits to a number before it writes the words. Of 191
  yes-quotes, 160 match inside the segment named and **2** match only in a
  different one. Worth about +0.002; not worth a capture.
- *Pressing harder on "copy the quote exactly".* 15% of quotes do not copy
  exactly — and they score **higher** (0.708 against 0.619). The longest-run
  matcher already absorbs the drift, and a paraphrase tends to mark a focused
  answer. There is nothing here to fix.

**Matching a quote across all segments when the named index looks off by one.**
This deletes the segment-scoped match, which `answering.py` documents as
load-bearing: it is the only thing stopping a phrase repeated later in the
consultation from dragging the span to the wrong mention. It aims at
`wrong-mention`, where the prompt route already scored 0.737 against 0.759.

### 14: The second pass — built, measured, and **declined** (2026-09-20)

**Do not turn this on.** Three full `tools/score.py` runs back to back, live
against qwen3:14b, with `ANSWER_DEADLINE` lifted so the budget could not
suppress the escalation — this measured the prompt, not the gate:

| run | accuracy | mean tIoU | score |
| --- | --- | --- | --- |
| second pass **off** | 0.990 | 0.621 | **0.768** |
| current `SELECT_TASK` | 0.990 | 0.589 | 0.749 |
| a redefined `SELECT_TASK` | 0.990 | 0.593 | 0.752 |

The `off` run reproduced the replayed 0.768 exactly, so the two below are the
prompt and not sampling noise. **Shown five overlapping extracts, the model
chooses worse than the heuristic that ranked them.** Accuracy never moves,
as designed: the second pass only ever touches spans.

The redefinition named the annotation convention instead of asking which
passage answers the question — *"choose the shortest passage that states the
answer by itself; one that carries a neighbouring sentence beyond that
statement is wrong, and so is one that cuts the statement short"*, which is the
actual choice, since the candidates are overlapping extracts of the same words
and two of them usually do answer it. It is worth **+0.003 against the current
wording and −0.016 against off**. Better wording, same verdict.

**Where that lands against the ceiling.** The gated ceiling is +0.017 and the
gated floor −0.112. Spread over the 61 questions the gate fires on, −0.016
overall is about 0.52 mean tIoU where the heuristic returns 0.609 and a perfect
chooser would return 0.702. So the model is well above a worst-case chooser and
below the heuristic — it is not noise, it is a worse ranker.

**What would have to change for this to be worth re-opening.** Not the wording:
two of them now bracket the result. Either a model that can rank overlapping
extracts, or a different question to ask it than "which of these five".

The design notes below are kept because they are what makes the decline
legible — the gate measurement in particular is a finding about where the
headroom sits, not about the second pass.

Ranking candidate passages today means counting how many of the question's
content words each carries. That signal is **spent**: the three constants
weighting it were all fitted and none moved. A *perfect* choice among the same
shortlist is worth **+0.068 mean tIoU, +0.041 of score** — the largest remaining
item on the evidence side, and about twenty times the inset.

So `SECOND_PASS=1` shows the model the shortlist the heuristic already built and
asks which passage answers the question. That is extraction from five numbered
lines, not the recall the earlier prompt experiment asked for and lost.

**The gate, and the awkward measurement in it.** Score closeness alone fires on
92% of questions, because candidates tie on shared-word count constantly. Adding
a minimum separation *in time* between the top two is what does the work:

| min gap | fires on | headroom reachable |
| --- | --- | --- |
| 0.0 s | 92% | 98% |
| 1.0 s | 88% | 95% |
| **2.0 s** | **32%** | **43%** |

**The cliff is the finding: the headroom is spread thinly across most questions,
not concentrated in a few ambiguous ones.** There is no gate that is both cheap
and complete. The shipped default (2.0 s) costs ~1.6 extra calls a conversation
— about 1.6 s hosted — for a *ceiling* of +0.018 score, and the model will not
reach its ceiling. Ungated is +0.041 at ~9 extra calls, or ~10 s hosted, which
is only affordable if the arrival-clock fix has genuinely recovered headroom.

Everything fails safe: a refusal, an unreadable reply, an out-of-range number or
too little time all keep the span the heuristic already chose. Seven tests pin
that, including that the selection prompt still opens with the shared header
(ADR-0001) and that a confident ranking is never escalated.

**Capturing it.** `tools/capture_replies.py` used to advance the question cursor
on every call, so a second call would have filed every later reply under the
wrong id and misaligned the whole capture *silently*. Fixed: selection replies
are filed as `<question_id>#select`. Note `tools/replay.py` cannot re-score a
second-pass run from frozen replies alone — the shortlist depends on the current
constants, so a recorded choice is only valid for the constants it was captured
under. Judge this one with `tools/score.py`, not replay.

    SECOND_PASS=1 ./.venv/bin/python tools/capture_replies.py --out diagnostics/replies_second.json
    SECOND_PASS=1 ./.venv/bin/python tools/score.py

## Still to do

### 12: More extent fitting, if anyone returns to it

**Blocked by:** Nothing. Read the protocol note below before re-running.

`tools/fit.py` searches the six constants in `answering.py` that decide how far a
passage runs. Two moved and are shipped: `MAX_RUN_SENTENCES` 4 -> 2 and
`REACH_SECONDS` 1.0 -> 1.25. The other four did not.

**The protocol is the part worth keeping.** Six parameters against 195 passages
from 39 conversations will always improve the number they were fitted on, so:

| test | delta | verdict |
| --- | --- | --- |
| in-sample, all six parameters | +0.0170 on the fit half | meaningless |
| single 25/14 split, holdout | +0.0066, CI touches zero | **no** |
| eight different 25/14 splits | +0.003 mean, one at -0.023 | **no** |
| nested leave-one-out CV, 39 folds | **+0.0131, CI +0.0023 to +0.0269** | **ship** |

The single split says no because 14 conversations is too small a holdout to
resolve an effect this size — one bad conversation swings it. Leave-one-out uses
all 39 as test data and resolves it. The signature that decided it: the six
parameter search shrank from +0.0170 in-sample to +0.0066 out, while the shipped
two-parameter change is **+0.0131 both in-sample and under nested CV** — no
shrinkage is what a real effect looks like.

Run it as `./.venv/bin/python tools/fit.py`. If you widen the grids or add a
parameter, re-run the nested CV before believing anything: an in-sample interval
that excludes zero means nothing when the parameters were chosen on that sample.

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

**Two routes into it are now closed.** Asking the model to choose among the
candidate sentence runs loses in both wordings (ticket 14), and re-wording what
the first pass quotes loses in all four directions measured (see the declines).
What is left is the *ranking* — `ANCHOR_WEIGHT`, `LENGTH_PENALTY`,
`TARGET_SECONDS` — and all three were fitted and none moved. Read that as a
warning about the size of what remains here, not as an invitation.

## Numbers worth not re-deriving

Ceilings, measured with `tools/score.py --ceiling` / `--ceiling-words` (no LLM,
returns in under a second):

| Span granularity | mean tIoU |
| --- | --- |
| Whole ASR segments | 0.521 |
| Whole sentences | 0.708 |
| Runs of up to four sentences | 0.815 |
| Best word run | 0.929 |
| Best word run, calibrated offset | 0.942 |
| Best word run, **WhisperX forced alignment** | 0.897 |
| ...calibrated | 0.936 |

**1.00 is not reachable.** A perfect system on today's ASR word timings caps at
about 0.963 final score. The gap is uniform quantization error, not a fixable
bias. **Finer timestamps do not raise it** — WhisperX forced alignment was built
and measured on 2026-09-20 and the ceiling goes *down*; see the section above.
Accuracy is worth +0.004 in total; do not spend time there.

How the annotated passages sit against our sentences, over all 195 (recompute
with the alignment check described in ticket 12 if the ASR or the guard changes):

| | |
| --- | --- |
| gold start within 0.25 s of a sentence start | 68.7% |
| gold end within 0.25 s of a sentence end | 81.0% |
| gold duration | median 2.88 s |
| covering sentence run | median 3.54 s |
| covering run *minus* gold, i.e. trimmable | median 0.20 s, p90 2.16 s |
| gold **longer** than its covering run, i.e. we under-reach | 15.9% |

Those two rows are why trimming was declined and why the pad was built and then
found worthless: the surplus is small and tail-shaped, not a systematic bias.

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
./.venv/bin/python tools/disagreements.py  # which direction the error runs
./.venv/bin/python tools/extent.py         # is "extract less" a lever? seconds
./.venv/bin/python tools/timings.py        # two word-timing sources, head to head
./.venv-align/bin/python tools/align.py    # re-time with WhisperX, ~70 s, own venv
TRANSCRIPTS_DIR=transcripts_aligned ./.venv/bin/python tools/replay.py
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
