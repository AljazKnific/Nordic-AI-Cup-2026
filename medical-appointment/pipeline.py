"""One conversation in, ten answers and their spans out.

Transcription and answering are injected so this can be exercised without a
model or audio; :mod:`example` wires the real ones in.

The contract is unforgiving and the failure is expensive: all three lists must
hold exactly one entry per question, in order. A list of the wrong length means
the body does not parse, and every question about the conversation is scored
wrong — ten marks, not one. Nothing in here may raise.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import request_log
from answering import Answerer, answer_conversation
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

Transcriber = Callable[[bytes], List[Dict[str, Any]]]

# The evaluator allows 60 s. Stop answering here instead, so the reply is on the
# wire before the budget expires: transcription is ~70% of the cost and scales
# with audio length, so on a long conversation or a slow host there may be very
# little left. A guessed answer is worth half a mark; a timeout is worth nothing
# and five in a row end the attempt.
#
# **This budget is measured from when the request arrived, not from when we
# started work on it.** A conversation is a 3-4 MB MP3, ~5 MB once base64'd,
# and uploading it is not free: on 2026-09-19 `conversation_sample_3.mp3` was
# answered in 35.0 s by this clock and still timed out at the evaluator's 60 s,
# so 25 s went somewhere this process could not see. Anything measured from
# inside `predict` is measuring the wrong 50 seconds.
REQUEST_BUDGET_SECONDS = float(os.environ.get('REQUEST_BUDGET', '50'))

# What one answer costs once the prompt prefix is warm: 1.2-1.5 s on an M4 Pro,
# 2.0-2.5 s on the hosted machine. The transcription deadline is derived from
# it rather than fixed, so transcription yields exactly when the questions still
# to come need the time. A fixed 34 s was never consistent with the rest of the
# budget -- 34 + 10 x 2.5 is 59 s against a 50 s allowance, which is precisely
# how the hosted run came to guess its last few questions.
SECONDS_PER_QUESTION = float(os.environ.get('SECONDS_PER_QUESTION', '2.5'))

# Seconds of *actual transcription work*, never less, whatever the budget says.
# This is a floor on work rather than a point on the clock on purpose: a slow
# upload can arrive with most of the 60 s already gone, and subtracting the
# questions' share from what is left would hand transcription zero seconds and
# return an empty transcript. Ten questions answered against no transcript at
# all score nothing; answered against the first fifteen seconds of the
# conversation they score something. Always come back with a transcript.
MIN_TRANSCRIBE_SECONDS = float(os.environ.get('MIN_TRANSCRIBE', '12'))


def _transcribe_deadline(started: float, questions: int) -> float:
    """When transcription must stop, so the questions still have time to run.

    Measured from now, not from arrival: by the time this is called the upload
    has already been paid for, and what matters is how much of the budget is
    still unspent.
    """
    now = time.perf_counter()
    remaining = REQUEST_BUDGET_SECONDS - (now - started)
    allowed = remaining - questions * SECONDS_PER_QUESTION
    return now + max(MIN_TRANSCRIBE_SECONDS, allowed)


def predict(
    request: ASRQuestionRequestDto,
    transcribe: Transcriber,
    answerer: Answerer,
    arrived: Optional[float] = None,
) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    ``arrived`` is a :func:`time.perf_counter` reading from when the request
    landed, before its body was read. Every budget here runs from it, so upload
    time is spent out of the same 60 s the evaluator is counting. Without it the
    clock starts once the body is already parsed, which is how a conversation
    answered in 35 s can still be scored as a timeout.
    """
    started = arrived if arrived is not None else time.perf_counter()
    questions: Sequence[str] = request.questions
    duration: Optional[float] = None
    # Write-only, and off unless REQUEST_LOG_DIR is set. The evaluation audio is
    # unseen, so this is the only way to read back what the ASR heard.
    record = request_log.Record(request.audio_filename)
    if request_log.enabled():
        answerer = record.recording(answerer)

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
        segments = transcribe(audio_bytes, deadline=_transcribe_deadline(
            started, len(questions)))
        transcription_seconds = time.perf_counter() - transcription_started

        answering_started = time.perf_counter()
        results = answer_conversation(
            segments, questions, answerer,
            deadline=started + REQUEST_BUDGET_SECONDS,
        )
        answering_seconds = time.perf_counter() - answering_started
        record.note(
            audio_seconds=duration,
            questions=list(questions),
            transcript=request_log.transcript_of(segments),
        )
    except Exception:
        # Whatever went wrong, a guess for every question beats no reply at all:
        # a coin toss is worth half a mark, an error is worth nothing, and
        # silence is worth less than that — five consecutive timeouts end the
        # attempt outright.
        logger.exception('Pipeline failed for %s; guessing every question', request.audio_filename)
        logger.info('TIMING %s: failed after %.1f s',
                    request.audio_filename, time.perf_counter() - started)
        record.note(failed=True, seconds=round(time.perf_counter() - started, 2))
        record.write()
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
        'TIMING %s: arrival-to-work %.1f s, transcribe %.1f s, answer %.1f s '
        '(%.1f s/question), total %.1f s',
        request.audio_filename,
        transcription_started - started,
        transcription_seconds,
        answering_seconds,
        answering_seconds / len(questions) if questions else 0.0,
        total,
    )

    record.note(
        answers=answers,
        evidence_start=starts,
        evidence_end=ends,
        timings={
            'arrival_to_work': round(transcription_started - started, 2),
            'transcribe': round(transcription_seconds, 2),
            'answer': round(answering_seconds, 2),
            'total': round(total, 2),
        },
    )
    record.write()

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
