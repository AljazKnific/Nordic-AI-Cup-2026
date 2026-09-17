"""One conversation in, ten answers and their spans out.

Transcription and answering are injected so this can be exercised without a
model or audio; :mod:`example` wires the real ones in.

The contract is unforgiving and the failure is expensive: all three lists must
hold exactly one entry per question, in order. A list of the wrong length means
the body does not parse, and every question about the conversation is scored
wrong — ten marks, not one. Nothing in here may raise.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from answering import Answerer, answer_conversation
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

Transcriber = Callable[[bytes], List[Dict[str, Any]]]


def predict(
    request: ASRQuestionRequestDto,
    transcribe: Transcriber,
    answerer: Answerer,
) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    started = time.perf_counter()
    questions: Sequence[str] = request.questions
    duration: Optional[float] = None

    try:
        audio_bytes = decode_audio(request.audio_base64)
        duration = audio_duration_seconds(audio_bytes)
        logger.info(
            '%s (%.1f s, %.1f MB): %d questions',
            request.audio_filename,
            duration if duration is not None else float('nan'),
            len(audio_bytes) / 1e6,
            len(questions),
        )

        transcription_started = time.perf_counter()
        segments = transcribe(audio_bytes)
        transcription_seconds = time.perf_counter() - transcription_started

        answering_started = time.perf_counter()
        results = answer_conversation(segments, questions, answerer)
        answering_seconds = time.perf_counter() - answering_started
    except Exception:
        # Whatever went wrong, a guess for every question beats no reply at all:
        # a coin toss is worth half a mark, an error is worth nothing, and
        # silence is worth less than that — five consecutive timeouts end the
        # attempt outright.
        logger.exception('Pipeline failed for %s; guessing every question', request.audio_filename)
        logger.info('TIMING %s: failed after %.1f s',
                    request.audio_filename, time.perf_counter() - started)
        return _guess(len(questions), duration)

    answers: List[bool] = []
    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    for answer, span in results:
        answers.append(bool(answer))
        starts.append(span[0] if span else None)
        ends.append(span[1] if span else None)

    total = time.perf_counter() - started
    logger.info(
        'TIMING %s: transcribe %.1f s, answer %.1f s (%.1f s/question), total %.1f s',
        request.audio_filename,
        transcription_seconds,
        answering_seconds,
        answering_seconds / len(questions) if questions else 0.0,
        total,
    )

    return ASRQuestionResponseDto(
        answers=answers, evidence_start=starts, evidence_end=ends,
    )


def _guess(count: int, duration: Optional[float] = None) -> ASRQuestionResponseDto:
    """A guess for every question, with the loosest span we can still justify.

    Pointing at the whole conversation scores about 0.04 rather than 0 -- poor,
    but strictly better than ``null``, which is guaranteed nothing. When even the
    duration is unknown there is nothing honest to point at.
    """
    span = (0.0, duration) if duration and duration > 0 else (None, None)
    return ASRQuestionResponseDto(
        answers=[True] * count,
        evidence_start=[span[0]] * count,
        evidence_end=[span[1]] * count,
    )
