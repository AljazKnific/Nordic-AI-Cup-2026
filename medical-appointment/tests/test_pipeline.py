"""Tests at the contract seam: predict.

A mistake here costs ten marks rather than one — a list of the wrong length
cannot be matched to the questions, so the whole conversation is scored wrong.
These assert the shape of the body, not the quality of the answers.
"""

import base64
import json
import time

import pytest

from dtos import ASRQuestionRequestDto
from pipeline import predict
from utils import validate_response

QUESTIONS = [f'question {i}?' for i in range(10)]


def request(questions=QUESTIONS):
    return ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b'not really an mp3').decode(),
        audio_filename='conversation_sample_17.mp3',
        questions=questions,
    )


def transcriber(segments):
    # Accepts the deadline the pipeline passes; the real transcriber uses it to
    # stop early rather than run the request past its budget.
    return lambda audio_bytes, deadline=None: segments


@pytest.fixture
def segments():
    return [
        {'index': 0, 'start': 0.0, 'end': 5.0, 'text': 'the daily dose is one hundred milligrams',
         'words': [{'start': 0.0, 'end': 1.0, 'word': 'the'},
                   {'start': 1.0, 'end': 2.0, 'word': 'daily'},
                   {'start': 2.0, 'end': 3.0, 'word': 'dose'},
                   {'start': 3.0, 'end': 4.0, 'word': 'is'},
                   {'start': 4.0, 'end': 5.0, 'word': 'one hundred milligrams'}],
         },
    ]


def answerer_returning(text):
    return lambda prompt, timeout=None: text


class TestContract:
    def test_response_passes_the_shipped_validator(self, segments):
        response = predict(
            request(),
            transcriber(segments),
            answerer_returning(json.dumps({'answer': 'yes', 'segment': 0, 'quote': 'daily dose'})),
        )

        validate_response(response, expected_count=10)

    def test_all_three_lists_match_the_question_count(self, segments):
        response = predict(request(), transcriber(segments), answerer_returning('no'))

        assert len(response.answers) == 10
        assert len(response.evidence_start) == 10
        assert len(response.evidence_end) == 10

    def test_a_no_carries_null_timestamps(self, segments):
        response = predict(request(), transcriber(segments), answerer_returning('no'))

        assert response.answers == [False] * 10
        assert response.evidence_start == [None] * 10
        assert response.evidence_end == [None] * 10

    def test_a_yes_carries_an_interval_that_ends_after_it_starts(self, segments):
        response = predict(
            request(),
            transcriber(segments),
            answerer_returning(json.dumps({'answer': 'yes', 'segment': 0, 'quote': 'daily dose'})),
        )

        for start, end in zip(response.evidence_start, response.evidence_end):
            assert start is not None and end is not None
            assert end > start

    def test_a_single_question_is_handled(self, segments):
        response = predict(request(['only one?']), transcriber(segments), answerer_returning('no'))

        validate_response(response, expected_count=1)


class TestBudget:
    """A late reply is worth nothing; a poor one on time is worth half a mark."""

    def test_slow_transcription_leaves_the_answerer_no_time_and_we_still_reply(self, segments):
        import pipeline

        asked = []

        def slow_transcribe(audio_bytes, deadline=None):
            # Stands in for a long conversation eating the whole budget in ASR.
            pipeline.REQUEST_BUDGET_SECONDS = -1.0
            return segments

        def answerer(prompt, timeout=None):
            asked.append(prompt)
            return 'no'

        original = pipeline.REQUEST_BUDGET_SECONDS
        try:
            response = predict(request(), slow_transcribe, answerer)
        finally:
            pipeline.REQUEST_BUDGET_SECONDS = original

        validate_response(response, expected_count=10)
        assert asked == []                      # the model was never called
        assert all(response.answers)            # every question still answered
        assert all(s is not None for s in response.evidence_start)


class TestNeverSilent:
    """Any reply beats none: five consecutive timeouts end the whole attempt."""

    def test_transcription_failure_still_returns_ten_answers(self):
        def exploding(audio_bytes, deadline=None):
            raise RuntimeError('ASR fell over')

        response = predict(request(), exploding, answerer_returning('no'))

        validate_response(response, expected_count=10)
        assert response.answers == [True] * 10

    def test_undecodable_audio_still_returns_ten_answers(self, segments):
        broken = ASRQuestionRequestDto(
            audio_base64='!!! not base64 !!!',
            audio_filename='conversation_sample_17.mp3',
            questions=QUESTIONS,
        )

        response = predict(broken, transcriber(segments), answerer_returning('no'))

        validate_response(response, expected_count=10)

    def test_an_empty_transcript_still_returns_ten_answers(self):
        response = predict(request(), transcriber([]), answerer_returning('no'))

        validate_response(response, expected_count=10)


class TestTheBudgetCoversTransport:
    """The evaluator's clock starts when it sends, so ours must too.

    A conversation answered in 35 s by a clock that started after the body was
    parsed was still scored as a timeout at 60 s: the upload was not free and
    was not being counted. These pin the arrival clock reaching the deadlines.
    """

    def test_time_before_arrival_is_spent_out_of_the_budget(self, segments):
        import pipeline

        deadlines = []

        def recording_answerer(prompt, timeout=None):
            deadlines.append(timeout)
            return 'no'

        late = time.perf_counter() - pipeline.REQUEST_BUDGET_SECONDS + 5.0
        predict(request(), transcriber(segments), recording_answerer, arrived=late)

        # Only ~5 s of the budget survives the upload, and the answerer is told.
        assert deadlines and deadlines[0] is not None
        assert deadlines[0] <= 5.0

    def test_a_request_that_arrived_too_long_ago_guesses_rather_than_stalls(self, segments):
        import pipeline

        asked = []

        def answerer(prompt, timeout=None):
            asked.append(prompt)
            return 'no'

        stale = time.perf_counter() - pipeline.REQUEST_BUDGET_SECONDS - 1.0
        response = predict(request(), transcriber(segments), answerer, arrived=stale)

        assert asked == []
        validate_response(response, expected_count=10)

    def test_transcription_yields_time_to_the_questions_still_to_come(self):
        import pipeline

        now = time.perf_counter()
        ten = pipeline._transcribe_deadline(now, 10) - now
        one = pipeline._transcribe_deadline(now, 1) - now

        # Ten questions reserve more than one does, and neither starves the ASR.
        assert one > ten
        assert ten >= pipeline.MIN_TRANSCRIBE_SECONDS
        assert ten <= pipeline.REQUEST_BUDGET_SECONDS

    def test_a_slow_upload_never_leaves_transcription_with_no_time(self):
        """An empty transcript scores nothing; a partial one scores something."""
        import pipeline

        arrived_long_ago = time.perf_counter() - pipeline.REQUEST_BUDGET_SECONDS
        allowed = pipeline._transcribe_deadline(arrived_long_ago, 10) - time.perf_counter()

        assert allowed >= pipeline.MIN_TRANSCRIBE_SECONDS - 0.1
