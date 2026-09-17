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
"""

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils import Span

logger = logging.getLogger(__name__)

Answerer = Callable[[str], str]

# Stop calling the model once a conversation has cost this much. Ten sequential
# calls at the client timeout would run to several minutes, and the budget is
# 60 s averaged across the attempt -- overrunning early silently costs marks on
# conversations that are then never sent. Past the deadline we guess, because a
# guess is worth half a mark and silence ends the attempt.
DEADLINE_SECONDS = float(os.environ.get('ANSWER_DEADLINE', '35'))

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


def _normalise(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', text.casefold()).strip()


def _words_of(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    return segment.get('words') or []


def segment_bounds(segment: Dict[str, Any]) -> Span:
    return (float(segment['start']), float(segment['end']))


def resolve_span(segment: Dict[str, Any], quote: Optional[str]) -> Span:
    """Narrow a segment to the quoted passage using its word timings.

    Falls back to the segment's own bounds whenever the quote is missing, does
    not match, or matches something implausibly long. The fallback is never
    ``None``: a loose span still scores, and ``None`` cannot.
    """
    words = _words_of(segment)
    if not quote or not words:
        return segment_bounds(segment)

    wanted = _normalise(quote).split()
    if not wanted:
        return segment_bounds(segment)

    spoken = [_normalise(word['word']) for word in words]

    # Longest run of the quote that appears contiguously in the segment. A model
    # that drops or adds a word at either end should still localise.
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
        logger.info('FALLBACK quote-unmatched in segment %s: %r', segment['index'], quote)
        return segment_bounds(segment)

    span = (float(words[best[0]]['start']), float(words[best[1]]['end']))
    if span[1] <= span[0]:
        return segment_bounds(segment)
    return span


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
) -> List[Tuple[bool, Optional[Span]]]:
    """Answer every question about one conversation, with a span for each yes.

    Never raises and never returns ``None`` for a yes. One bad reply costs one
    question; the conversation's other nine are unaffected.
    """
    if not segments:
        logger.warning('No segments to answer against; guessing yes with no span')
        return [(True, None) for _ in questions]

    header = build_transcript_header(segments)
    results: List[Tuple[bool, Optional[Span]]] = []
    started = time.perf_counter()

    for question in questions:
        if time.perf_counter() - started > DEADLINE_SECONDS:
            logger.warning('FALLBACK past the %.0f s deadline; guessing: %s',
                           DEADLINE_SECONDS, question)
            segment = _best_keyword_segment(segments, question)
            results.append((True, segment_bounds(segment)))
            continue

        try:
            answer, index, quote = parse_reply(answerer(build_prompt(header, question)))
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

        results.append((True, resolve_span(segment, quote)))

    return results
