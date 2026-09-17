"""Tests at the contract seam: predict.

A mistake here costs ten marks rather than one — a list of the wrong length
cannot be matched to the questions, so the whole conversation is scored wrong.
These assert the shape of the body, not the quality of the answers.
"""

import base64
import json

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
    return lambda audio_bytes: segments


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
    return lambda prompt: text


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


class TestNeverSilent:
    """Any reply beats none: five consecutive timeouts end the whole attempt."""

    def test_transcription_failure_still_returns_ten_answers(self):
        def exploding(audio_bytes):
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
