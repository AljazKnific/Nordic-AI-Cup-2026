"""Answering ten questions against one transcript, and locating the passage.

The seam is :func:`answer_conversation`. The answerer is injected rather than
reached for, so span logic, parsing and the fallback chain are all testable
without Ollama.

Two things in here are load-bearing and easy to break by accident:

**The shared prefix** (ADR-0001). Every prompt opens with a byte-identical
transcript header and puts its task *after* it, so Ollama reuses the cached
attention state across the ten questions. Moving the question ahead of the
transcript costs ~70 s per conversation and changes no output, so nothing will
tell you it happened.

**Segment-scoped quote matching.** A quote is matched only inside the segment the
model named. A phrase repeated elsewhere in the conversation is therefore
unmatchable, which is what stops a patient echoing the doctor from dragging the
span to the wrong mention.

**Sentences, not segments, are the unit we point at.** The ASR divides speech on
silence, so its segments cut sentences in half and a span trimmed inside one is
routinely half a passage. Once located, a span is rounded out to the sentences
it lies in -- which may cross a segment boundary, though the *match* never does.
"""

import json
import logging
import os
import re
import time
from typing import (
    Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple,
)

from utils import Span

logger = logging.getLogger(__name__)

# The answerer takes a prompt and the seconds of budget left, and returns the
# model's raw reply. The budget is part of the seam on purpose: a deadline
# checked only between calls cannot stop a single slow call from overrunning it.
Answerer = Callable[..., str]

# Stop calling the model once a conversation has cost this much. Ten sequential
# calls at the client timeout would run to several minutes, and the budget is
# 60 s averaged across the attempt -- overrunning early silently costs marks on
# conversations that are then never sent. Past the deadline we guess, because a
# guess is worth half a mark and silence ends the attempt.
#
# The deadline is enforced twice: once here, deciding whether to ask at all,
# and once inside the call, bounding how long we wait. The first alone leaves a
# call that starts just inside the deadline free to return long after it.
DEADLINE_SECONDS = float(os.environ.get('ANSWER_DEADLINE', '35'))

# --- The second pass -------------------------------------------------------
#
# Off by default. Ranking candidate passages today means counting how many of
# the question's content words each one carries, and that signal is spent: the
# three constants weighting it (ANCHOR_WEIGHT, LENGTH_PENALTY, TARGET_SECONDS)
# were all fitted and none of them moved. Meanwhile a *perfect* choice among the
# same shortlist is worth +0.068 mean tIoU, or +0.041 of final score -- the
# largest remaining item on the evidence side.
#
# So the second pass shows the model the shortlist the heuristic already built
# and asks which passage answers the question. That is an extraction from five
# numbered lines, not the recall the earlier prompt experiment asked for and
# lost (0.737 against 0.759).
#
# **It is gated, because it is not free.** An ungated second call per question
# adds 20-25 s to a hosted conversation, and timeouts have cost more than every
# modelling change here put together. It fires only when the top two candidates
# are within SELECT_MARGIN of each other -- if the heuristic is confident there
# is nothing to ask -- and only when there is time to spare.
SECOND_PASS = os.environ.get('SECOND_PASS', '') not in ('', '0', 'false')

# How close the top two have to be before the choice is worth a call. Score
# units are "question content words shared", so 1.0 means a clear winner is left
# alone and a tie or near-tie is escalated.
SELECT_MARGIN = float(os.environ.get('SELECT_MARGIN', '1.0'))

# ...and how far apart they have to be *in time*. A near-tie between two spans
# that name nearly the same seconds is not a question worth asking. This is the
# condition that does the gating work: on score alone the gate fires on 92% of
# questions, because candidates tie on shared-word count constantly.
#
# The measured trade, over the 39 supplied conversations, where "headroom" is
# the +0.068 mean tIoU a perfect choice among the shortlist would win:
#
#   min gap   fires on   headroom reachable
#   0.0 s        92%           98%
#   1.0 s        88%           95%
#   2.0 s        31%           43%
#
# That cliff is the finding: the headroom is spread thinly across most
# questions rather than concentrated in a few ambiguous ones, so there is no
# gate that is both cheap and complete. 2.0 s is the honest compromise -- about
# three extra calls a conversation for a little under half the available gain.
SELECT_MIN_GAP = float(os.environ.get('SELECT_MIN_GAP', '2.0'))

# Never start a second call with less than this left: it must not be the thing
# that pushes a conversation over the budget.
SELECT_MIN_SECONDS = float(os.environ.get('SELECT_MIN_SECONDS', '6'))

# The shortlist shown to the model. Long lists cost tokens and invite the model
# to pick from the middle; five is the median number of candidates anyway.
SELECT_SHORTLIST = int(os.environ.get('SELECT_SHORTLIST', '5'))

SELECT_TASK = """
Which passage below is the one that answers this question?

Question: %s

%s
Reply with the number of that passage only, as a plain integer.
"""

TASK = """
Answer the question below using only the consultation above.

The questions are adversarial: many are near-misses on something the
consultation does establish -- the right drug at the wrong dose, the right
course at the wrong length. Compare every number, unit, drug name and duration
against the transcript before answering yes. If the question differs from the
transcript in any detail, the answer is no. If the subject never comes up at
all, the answer is no.

Reply with JSON only:
{"answer": "yes" or "no", "segment": <the number of the segment you read the answer off, as a plain integer with no brackets>, "quote": "<the exact words from that segment that answer it>"}

For "no", use {"answer": "no", "segment": null, "quote": null}.
The quote must be copied exactly from the segment, not paraphrased.

Question: """


def build_transcript_header(segments: Sequence[Dict[str, Any]]) -> str:
    """The identical prefix every question in a conversation is asked against.

    Must not vary per question -- see ADR-0001.
    """
    lines = ['Below is a transcript of a medical consultation, in numbered segments.', '']
    for segment in segments:
        lines.append(f"[{segment['index']}] {segment['start']}-{segment['end']}  {segment['text']}")
    lines.append('')
    return '\n'.join(lines)


def build_prompt(header: str, question: str) -> str:
    """Header first, always. The task and the question go after it."""
    return header + TASK + question + '\n'


def build_selection_prompt(
    header: str, question: str, candidates: Sequence['Candidate'],
) -> str:
    """The second pass's prompt. Same header, so the cached prefix still holds.

    Sharing the header is the whole reason this is affordable: the transcript
    is already in Ollama's cache from the first call, so only the short tail
    below is evaluated. Putting anything before the header would cost ~70 s a
    conversation and nothing would report it (ADR-0001).
    """
    lines = '\n'.join(
        f'[{n + 1}] {candidate.text}' for n, candidate in enumerate(candidates))
    return header + SELECT_TASK % (question, lines)


def ambiguous(candidates: Sequence['Candidate']) -> bool:
    """Whether the top two are worth a call: close in score, apart in time."""
    if len(candidates) < 2:
        return False
    if (candidates[0].score - candidates[1].score) >= SELECT_MARGIN:
        return False
    first, second = candidates[0].span, candidates[1].span
    apart = abs(first[0] - second[0]) + abs(first[1] - second[1])
    return apart > SELECT_MIN_GAP


def parse_selection(text: str, count: int) -> Optional[int]:
    """The 1-based number the model picked, as a 0-based index, or ``None``.

    Anything unreadable or out of range returns ``None``, and the caller keeps
    the heuristic's own choice. A second pass that cannot be understood must
    cost nothing rather than a span.
    """
    if not text:
        return None
    match = re.search(r'\d+', text)
    if not match:
        return None
    chosen = int(match.group(0)) - 1
    return chosen if 0 <= chosen < count else None


def _normalise(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', text.casefold()).strip()


def _words_of(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    return segment.get('words') or []


def segment_bounds(segment: Dict[str, Any]) -> Span:
    return (float(segment['start']), float(segment['end']))


# Gold passages run a median of 2.9 s and are written as sentences, while the
# ASR divides speech into ~3 s segments that cut sentences in half: 68 of the
# 195 annotated passages cross a segment boundary. Pointing at whole segments is
# worth only 0.521 mean tIoU where whole sentences are worth 0.708 and runs of
# them 0.815, so the unit we return is the sentence, not the segment and not the
# model's quote. The quote says *where*; the sentences say *how far*.
#
# This lets a span cross a segment boundary, which the segment-scoped *match*
# never does. The two are different things. The quote is still matched only
# inside the segment the model named, so a phrase repeated elsewhere in the
# conversation still cannot drag the span to the wrong mention; all this does is
# stop the segment boundary truncating the sentence the match landed in.
_SENTENCE_END = ('.', '?', '!')

# Titles the ASR writes with a trailing stop, which is not the end of a
# sentence. "Dr." alone accounts for 38 of the 2,132 boundaries the ASR emits
# over the 39 supplied conversations, and every one of them fractures a sentence
# that the gold annotation keeps whole.
#
# Titles *only*, and deliberately so. The other abbreviation-shaped tokens in
# these transcripts are "no." (14) and "am." (4), and in a consultation both are
# overwhelmingly genuine sentence ends -- "No." is an answer to a question, and
# "I am." closes one. Guarding those would manufacture run-on sentences, which
# cost far more than the split they prevent. Unit abbreviations ("mg.", "ml.")
# appear zero times here, so adding them would be speculation, not a fix.
_ABBREVIATIONS = frozenset({'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.'})

# A sentence merely abutting the span is not part of it.
_TOUCH_SECONDS = 0.05

# No real sentence runs this long: the longest across the 39 supplied
# conversations is 8.6 s and the 99th percentile is 5.9 s. A "sentence" past
# this is the ASR having emitted no punctuation, and rounding a span out to one
# would hand back most of the conversation -- far worse than the segment we
# started from. So past this we keep the span we were given.
MAX_PASSAGE_SECONDS = 15.0

# A model asked to quote what it read hands back the sentence it was reading and
# often its neighbours, so among the sentences near a located span we prefer the
# ones carrying the question's own terms, discounting candidates that run well
# past the length of a typical passage.
TARGET_SECONDS = 3.0
LENGTH_PENALTY = 0.2

# How far past the located span a passage may still reach, in seconds.
#
# The reason first given for this was wrong, and measuring it is what found the
# error: an annotated passage is *not* usually "the exchange, doctor's question
# through patient's answer". **72% of gold passages are exactly one sentence**,
# 19% are two, and only 2% of the one-sentence ones open with a question. The
# reach is not buying back exchanges, because there are hardly any.
#
# What it does buy is tolerance: the quote the model returns rarely lines up
# with a sentence boundary, so a window that reaches a little past it keeps the
# right sentence in the candidate set. That is why the value survived fitting
# (all 39 folds chose 1.25) while the theory behind it did not.
#
# 1.25 rather than 1.0 because that is what fitting chose: see `tools/fit.py`,
# where all 39 leave-one-conversation-out folds picked it independently.
# Symmetric, and measured twice. Splitting it into a backward and a forward
# reach loses under nested CV (-0.0083, CI -0.0215 to +0.0000), and 37 of the 39
# folds chose the two halves equal anyway -- in both directions: reaching
# *further* forward was rejected earlier, and reaching *less* forward was
# rejected on 2026-09-19. See `docs/tickets.md`.
REACH_SECONDS = 1.25

# Weight on how much of the located span a candidate still covers. Reaching
# outward needs a counterweight, or a neighbouring sentence with one question
# word in it outscores the sentence the quote was actually read from. Small on
# purpose: it breaks ties towards the quote rather than overruling the terms.
ANCHOR_WEIGHT = 0.5

# Passages run to at most a few sentences, and capping the run stops a candidate
# growing to cover a whole exchange plus its neighbours.
#
# Two, not four. Four was chosen as "well past the longest annotated passage",
# which was the wrong question: what matters is not whether a long run is ever
# right but how often the extra length is wrong. A gold passage runs a median of
# 2.88 s against a covering sentence run of 3.54 s, so the room a fourth
# sentence buys is nearly always room to overshoot. Fitted, and every one of the
# 39 leave-one-out folds chose 2 -- see `tools/fit.py`.
MAX_RUN_SENTENCES = 2

# A constant widening applied to both edges of the finished passage. Sentence
# bounds and annotated bounds do not coincide exactly -- an annotation tends to
# open a little before the first word and close a little after the last -- and
# on the 39 supplied conversations the covering sentence run is *shorter* than
# the gold passage in 16% of cases. One constant buys those back without
# touching which sentences were chosen.
#
# Symmetric on purpose. A separate forward and backward reach was measured and
# is worth +0.002 with a bootstrap interval straddling zero; splitting this into
# two parameters would be re-buying that result with a new name.
#
# **Negative: the passage is inset, not widened.** The first fit of this chose
# 0.0, but only because the grid ran from zero upwards -- widening was on it and
# shrinking was not. With negatives on the grid all 39 leave-one-out folds chose
# -0.05, worth +0.0023 mean tIoU with a CI of +0.0004 to +0.0044.
#
# Small, and that is the finding rather than a disappointment: where we cover
# the gold passage and overshoot, the sentence is a median 0.38 s too wide, but
# an inset big enough to matter cuts into the passage. -0.10 is already worse
# than nothing and -0.30 costs 0.06. The overshoot is not a margin to be shaved.
PAD_SECONDS = -0.05

# Too common to say anything about which sentence answers a question. Kept small
# and generic on purpose: a medical lexicon here would be fitted to the 39
# supplied conversations, and the evaluation drugs are unseen.
_STOPWORDS = frozenset("""
a an and are as at be been being but by did do does for from had has have he
her him his how i if in into is it its me my not of on or our she should so
that the their them then there these they this to too was we were what when
which who will with would you your patient
""".split())


class Sentence(NamedTuple):
    start: float
    end: float
    text: str
    # Kept so a passage can be trimmed inside the sentence it lands in. The
    # word timings are already paid for by the ASR and already good enough to
    # place a passage at 0.929 mean tIoU; nothing here needs finer ones.
    words: tuple = ()


def _content_words(text: str) -> set:
    return {
        word for word in _normalise(text).split()
        if word not in _STOPWORDS and len(word) > 1
    }


def sentences_of(segments: Sequence[Dict[str, Any]]) -> List[Sentence]:
    """The conversation's sentences, from the punctuation the ASR already emits.

    Built from the word stream rather than per segment, because a sentence
    routinely runs across a segment boundary -- which is most of why segment
    bounds localise so poorly.
    """
    found: List[Sentence] = []
    current: List[Dict[str, Any]] = []
    for segment in segments:
        for word in _words_of(segment):
            current.append(word)
            if _ends_sentence(word):
                found.append(_sentence(current))
                current = []
    if current:
        found.append(_sentence(current))
    return found


def _ends_sentence(word: Dict[str, Any]) -> bool:
    """Whether this word closes a sentence, or merely carries a trailing stop."""
    token = str(word.get('word', '')).strip()
    if not token.endswith(_SENTENCE_END):
        return False
    return token.casefold() not in _ABBREVIATIONS


def _sentence(words: Sequence[Dict[str, Any]]) -> Sentence:
    return Sentence(
        float(words[0]['start']),
        float(words[-1]['end']),
        ' '.join(str(word.get('word', '')) for word in words),
        tuple((float(w['start']), float(w['end']), str(w.get('word', '')))
              for w in words),
    )


# Words a sentence opens with that the annotation does not include. A gold
# passage starts a median of 0.14 s after the sentence carrying it, and where we
# cover gold and overshoot, the sentence is a median 0.38 s too wide -- about
# what one of these costs. Openers only: the same word mid-sentence is ordinary
# speech, and trailing filler is rarer here than leading.
_OPENERS = frozenset("""
so ok okay right well now um uh erm ah oh yeah yes no and but then alright
""".split())

# **Measured, and it loses.** -0.0056 mean tIoU with the interval entirely
# below zero, and all 39 folds chose it off. The reasoning was sound -- a gold
# passage starts a median 0.14 s after its sentence, about what "So," costs --
# but the annotation evidently keeps those openers more often than it drops
# them. Left in place and on the fit grid so the result can be re-checked in
# seconds instead of rebuilt; do not turn it on without new evidence.
TRIM_OPENERS = 0


def _trim_openers(passage: Span, sentences: Sequence[Sentence]) -> Span:
    """Drop discourse openers from the front of the sentence a passage starts in.

    Only ever moves the start forward, never past the sentence's own last word,
    and only when the passage genuinely begins at a sentence boundary -- a span
    that already starts mid-sentence has nothing to trim.
    """
    if not TRIM_OPENERS:
        return passage
    for sentence in sentences:
        if abs(sentence.start - passage[0]) > 0.05 or not sentence.words:
            continue
        for start, _end, word in sentence.words[:-1]:
            if _normalise(word) in _OPENERS:
                continue
            if start <= passage[0] + 0.01 or start >= passage[1]:
                return passage
            return (start, passage[1])
        return passage
    return passage


def _overlap(span: Span, sentence: Sentence) -> float:
    return min(span[1], sentence.end) - max(span[0], sentence.start)


def to_passage(
    span: Span, sentences: Sequence[Sentence], question: Optional[str] = None,
) -> Span:
    """Round a located span out to the sentences that carry the answer.

    Every span we return goes through here, so what we point at is always a
    whole number of sentences. Returns the span untouched when it lies in no
    sentence at all, or when the sentences it lies in are too long to be real --
    both of which mean the ASR emitted no usable punctuation here.

    Candidates are runs of neighbouring sentences, taken from a window that
    reaches ``REACH_SECONDS`` past the located span at either end, and scored on
    the question's own terms against how much of the located span they keep.
    Reaching outward is what lets a passage that opens with the doctor's
    question and closes with the patient's answer be returned whole, when the
    model quoted only one half of it.
    """
    candidates = passage_candidates(span, sentences, question)
    if candidates is None:
        return _finish(span, sentences)
    if not candidates:
        touched = [s for s in sentences if _overlap(span, s) > _TOUCH_SECONDS]
        return _plausible((touched[0].start, touched[-1].end), span, sentences)
    return _plausible(candidates[0].span, span, sentences)


class Candidate(NamedTuple):
    """One run of neighbouring sentences, and how well it scored."""
    score: float
    span: Span
    text: str


def passage_candidates(
    span: Span, sentences: Sequence[Sentence], question: Optional[str] = None,
) -> Optional[List[Candidate]]:
    """Every run of sentences worth considering, best first.

    Split out of :func:`to_passage` so the ranking can be inspected, and so a
    second pass can be offered the same shortlist the heuristic sees rather than
    a different one. Returns ``None`` when the span lies in no sentence at all,
    and an empty list when there is nothing to rank between.
    """
    window = (span[0] - REACH_SECONDS, span[1] + REACH_SECONDS)
    near = [s for s in sentences if _overlap(window, s) > _TOUCH_SECONDS]
    touched = [s for s in sentences if _overlap(span, s) > _TOUCH_SECONDS]
    if not touched:
        return None

    asked = _content_words(question) if question else set()
    if not asked or len(near) < 2:
        return []

    located = max(span[1] - span[0], 0.1)
    found: List[Candidate] = []
    for first in range(len(near)):
        for last in range(first, min(first + MAX_RUN_SENTENCES, len(near))):
            candidate = (near[first].start, near[last].end)
            if candidate[1] - candidate[0] > MAX_PASSAGE_SECONDS:
                continue
            text = ' '.join(s.text for s in near[first:last + 1])
            shared = len(asked & _content_words(text))
            kept = max(0.0, min(span[1], candidate[1])
                       - max(span[0], candidate[0])) / located
            score = (shared + ANCHOR_WEIGHT * kept
                     - LENGTH_PENALTY * max(
                         0.0, (candidate[1] - candidate[0]) - TARGET_SECONDS))
            found.append(Candidate(score, candidate, text.strip()))
    # Stable: equal scores keep the order they were generated in, which is
    # earliest-and-shortest first, so the default choice does not move.
    found.sort(key=lambda c: -c.score)
    return found


def _plausible(passage: Span, span: Span,
               sentences: Sequence[Sentence] = ()) -> Span:
    """The rounded-out passage, unless it is too long to be one."""
    if passage[1] - passage[0] > MAX_PASSAGE_SECONDS:
        logger.info('FALLBACK sentence longer than %.0f s; keeping the located span',
                    MAX_PASSAGE_SECONDS)
        return _finish(span, sentences)
    return _finish(passage, sentences)


def _padded(passage: Span) -> Span:
    """The passage widened by the constant offset -- or narrowed, if negative.

    A negative pad insets both edges, which is the direction the data actually
    asks for: where we cover the gold passage and overshoot, the sentence we
    return is a median 0.38 s wider than it. Never inset past the midpoint.
    """
    if not PAD_SECONDS:
        return passage
    start, end = passage[0] - PAD_SECONDS, passage[1] + PAD_SECONDS
    if start >= end:
        return passage
    return (max(0.0, start), end)


def _finish(passage: Span, sentences: Sequence[Sentence]) -> Span:
    """Everything applied to a passage after its sentences have been chosen."""
    return _padded(_trim_openers(passage, sentences))


def locate_quote(segment: Dict[str, Any], quote: Optional[str]) -> Optional[Span]:
    """Where a quote lies inside one segment, or ``None`` if it does not.

    Matched only within ``segment``, which is what stops a phrase repeated
    elsewhere dragging the span to the wrong mention. The match is the longest
    run of the quote's words appearing contiguously, so a model that drops or
    adds a word at either end still localises.

    Split out of :func:`resolve_span` so a prompt experiment can anchor on
    something other than one quote -- two phrases marking each end, say --
    without reimplementing the matcher and quietly changing it.
    """
    words = _words_of(segment)
    wanted = _normalise(quote).split() if quote else []
    if not words or not wanted:
        return None

    spoken = [_normalise(word['word']) for word in words]
    best: Optional[Tuple[int, int]] = None
    for length in range(len(wanted), 0, -1):
        for offset in range(len(wanted) - length + 1):
            needle = wanted[offset:offset + length]
            for start in range(len(spoken) - length + 1):
                if spoken[start:start + length] == needle:
                    best = (start, start + length - 1)
                    break
            if best:
                break
        if best:
            break

    if best is None:
        # Counted, not silenced: how often this fires is the measurement that
        # says whether the quote path is worth keeping.
        logger.info('FALLBACK quote-unmatched in segment %s: %r',
                    segment['index'], quote)
        return None

    candidate = (float(words[best[0]]['start']), float(words[best[1]]['end']))
    return candidate if candidate[1] > candidate[0] else None


def resolve_span(
    segment: Dict[str, Any],
    quote: Optional[str],
    sentences: Sequence[Sentence] = (),
    question: Optional[str] = None,
) -> Span:
    """Locate the passage a quote was read off, inside one named segment.

    The match is rounded out to whole sentences by :func:`to_passage`. Falls
    back to the segment's own bounds whenever the quote is missing or does not
    match. The fallback is never ``None``: a loose span still scores, and
    ``None`` cannot.
    """
    span = locate_quote(segment, quote)
    if span is None:
        span = segment_bounds(segment)
    return to_passage(span, sentences, question)


def _find_segment(segments: Sequence[Dict[str, Any]], index: Any) -> Optional[Dict[str, Any]]:
    # The model echoes the prompt's [n] notation, so the index arrives as [6],
    # "[6]" or 6 depending on the reply. Losing this costs the span but not the
    # answer, which makes it invisible in accuracy and expensive in tIoU.
    if isinstance(index, (list, tuple)):
        index = index[0] if len(index) == 1 else None
    if isinstance(index, str):
        digits = re.search(r'\d+', index)
        index = digits.group(0) if digits else None

    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    for segment in segments:
        if segment['index'] == index:
            return segment
    return None


def _best_keyword_segment(
    segments: Sequence[Dict[str, Any]], question: str
) -> Dict[str, Any]:
    """The last-resort span: whichever segment shares most words with the question."""
    asked = set(_normalise(question).split())
    return max(
        segments,
        key=lambda segment: len(asked & set(_normalise(segment['text']).split())),
    )


def parse_reply(text: str) -> Tuple[Optional[bool], Any, Optional[str]]:
    """Read an answer, a segment index and a quote out of a model reply.

    Layered deliberately: strict JSON, then a bare yes/no, then nothing. Returns
    ``(None, None, None)`` when even the answer could not be read, which the
    caller turns into a guess rather than an error.
    """
    if not text:
        return None, None, None

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            answer = payload.get('answer')
            if isinstance(answer, str):
                answer = answer.strip().casefold().startswith('y')
            if isinstance(answer, bool):
                return answer, payload.get('segment'), payload.get('quote')
        except (ValueError, AttributeError):
            pass

    leading = re.match(r'\W*(yes|no)\b', text.strip(), re.IGNORECASE)
    if leading:
        logger.info('FALLBACK reply was not JSON; read a bare yes/no')
        return leading.group(1).casefold() == 'yes', None, None

    logger.info('FALLBACK reply unreadable: %r', text[:120])
    return None, None, None


def answer_conversation(
    segments: Sequence[Dict[str, Any]],
    questions: Sequence[str],
    answerer: Answerer,
    deadline: Optional[float] = None,
) -> List[Tuple[bool, Optional[Span]]]:
    """Answer every question about one conversation, with a span for each yes.

    Never raises and never returns ``None`` for a yes. One bad reply costs one
    question; the conversation's other nine are unaffected.
    """
    if not segments:
        logger.warning('No segments to answer against; guessing yes with no span')
        return [(True, None) for _ in questions]

    header = build_transcript_header(segments)
    sentences = sentences_of(segments)
    results: List[Tuple[bool, Optional[Span]]] = []
    # An absolute deadline from the caller accounts for time already spent on
    # transcription, which is the larger and more variable half. Without one,
    # fall back to a budget for answering alone.
    if deadline is None:
        deadline = time.perf_counter() + DEADLINE_SECONDS

    for question in questions:
        if time.perf_counter() > deadline:
            logger.warning('FALLBACK past the deadline; guessing: %s', question)
            segment = _best_keyword_segment(segments, question)
            results.append(
                (True, to_passage(segment_bounds(segment), sentences, question)))
            continue

        try:
            # What is left of the budget, so a slow or cold call fails inside it
            # rather than running past it. The check above decides whether to
            # ask; this decides how long we are willing to wait for the answer.
            remaining = deadline - time.perf_counter()
            reply = answerer(build_prompt(header, question), timeout=remaining)
            answer, index, quote = parse_reply(reply)
        except Exception:
            logger.exception('Answerer failed; guessing for: %s', question)
            answer, index, quote = None, None, None

        if answer is False:
            results.append((False, None))
            continue

        if answer is None:
            logger.warning('Unreadable reply; guessing yes for: %s', question)

        segment = _find_segment(segments, index)
        if segment is None:
            logger.info('FALLBACK no usable segment index (%r) for: %s', index, question)
            segment = _best_keyword_segment(segments, question)

        span = resolve_span(segment, quote, sentences, question)
        if SECOND_PASS:
            span = _reconsider(
                segment, quote, sentences, question, header, answerer,
                deadline, span,
            )
        results.append((True, span))

    return results


def _reconsider(
    segment: Dict[str, Any],
    quote: Optional[str],
    sentences: Sequence[Sentence],
    question: str,
    header: str,
    answerer: Answerer,
    deadline: float,
    fallback: Span,
) -> Span:
    """Ask the model to choose among the candidates, when the ranking is close.

    Never raises and never returns ``None``: every way this can go wrong ends
    with ``fallback``, the span the heuristic already chose. The second pass is
    allowed to improve a span and never to cost one.
    """
    remaining = deadline - time.perf_counter()
    if remaining < SELECT_MIN_SECONDS:
        logger.info('SECOND-PASS skipped, %.1f s left', remaining)
        return fallback

    located = resolve_span(segment, quote, (), question)
    candidates = passage_candidates(located, sentences, question)
    if not candidates or not ambiguous(candidates):
        return fallback

    shortlist = candidates[:SELECT_SHORTLIST]
    try:
        reply = answerer(
            build_selection_prompt(header, question, shortlist),
            timeout=remaining,
        )
    except Exception:
        logger.info('SECOND-PASS call failed; keeping the heuristic span')
        return fallback

    chosen = parse_selection(reply, len(shortlist))
    if chosen is None:
        logger.info('SECOND-PASS unreadable reply %r; keeping the heuristic span',
                    (reply or '')[:60])
        return fallback

    picked = _plausible(shortlist[chosen].span, located, sentences)
    logger.info('SECOND-PASS chose [%d] of %d', chosen + 1, len(shortlist))
    return picked
