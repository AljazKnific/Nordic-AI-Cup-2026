"""Tests at the primary seam: answer_conversation.

The answerer is injected, so every one of these runs without Ollama, without
audio and without a model — which is the point of the seam. It also lets us feed
the malformed replies a real model produces only occasionally, and which cost ten
marks when they are mishandled.
"""

import json

import pytest

import answering


def inset_of(bounds):
    """Bounds as they come back once the constant inset has been applied.

    Every returned passage is moved in by ``PAD_SECONDS`` at each edge. Tests
    that pin exact bounds are asserting *which* sentence was chosen, so they
    say so through here rather than hard-coding the offset in each one.
    """
    inset = answering.PAD_SECONDS
    return (round(max(0.0, bounds[0] - inset), 2), round(bounds[1] + inset, 2))

from answering import answer_conversation, build_transcript_header


def segment(index, start, end, text):
    """A segment with one word per token, evenly spaced across its duration."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return {
        'index': index,
        'start': start,
        'end': end,
        'text': text,
        'words': [
            {
                'start': round(start + i * step, 2),
                'end': round(start + (i + 1) * step, 2),
                'word': token,
            }
            for i, token in enumerate(tokens)
        ],
    }


@pytest.fixture
def segments():
    return [
        segment(0, 0.0, 6.0, 'Good morning I am Doctor Norgaard could I have your name'),
        segment(1, 6.0, 12.0, 'I need my prescriptions renewed the asthma medicine please'),
        segment(2, 12.0, 20.0, 'We will renew the Airomir and the Esomeprazole today that is three in total'),
        segment(3, 20.0, 26.0, 'Take the daily dose of one hundred milligrams after a meal'),
    ]


def replies(*canned):
    """An answerer returning canned replies in order, recording its prompts."""
    prompts = []
    queue = list(canned)

    def answerer(prompt, timeout=None):
        prompts.append(prompt)
        return queue.pop(0)

    answerer.prompts = prompts
    return answerer


def reply(answer, segment_index=None, quote=None):
    return json.dumps({'answer': answer, 'segment': segment_index, 'quote': quote})


class TestAnswers:
    def test_yes_reply_answers_yes(self, segments):
        answerer = replies(reply('yes', 3, 'one hundred milligrams'))
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span is not None

    def test_no_reply_answers_no_with_no_span(self, segments):
        answerer = replies(reply('no'))
        [(answer, span)] = answer_conversation(segments, ['Is the dose 200 mg?'], answerer)

        assert answer is False
        assert span is None

    def test_every_question_gets_an_entry_in_order(self, segments):
        answerer = replies(reply('yes', 3, 'one hundred'), reply('no'), reply('yes', 2, 'Airomir'))
        results = answer_conversation(segments, ['a?', 'b?', 'c?'], answerer)

        assert [answer for answer, _ in results] == [True, False, True]


class TestSpans:
    def test_span_is_trimmed_to_the_quote_not_the_segment(self, segments):
        answerer = replies(reply('yes', 3, 'one hundred milligrams'))
        [(_, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        # Segment 3 runs 20.0-26.0; the quote sits inside it.
        assert 20.0 < span[0]
        assert span[1] < 26.0
        assert span[1] - span[0] < 6.0

    def test_unmatched_quote_falls_back_to_segment_bounds(self, segments):
        answerer = replies(reply('yes', 3, 'words that were never spoken here'))
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span == inset_of((20.0, 26.0))

    def test_quote_is_matched_only_inside_the_named_segment(self, segments):
        """The duplicate-mention guarantee: 'the' appears in several segments."""
        answerer = replies(reply('yes', 2, 'Airomir and the Esomeprazole'))
        [(_, span)] = answer_conversation(segments, ['Was Airomir renewed?'], answerer)

        assert 12.0 <= span[0] and span[1] <= 20.0

    def test_a_span_never_escapes_the_segment_it_was_read_from(self, segments):
        """Words come from the segment, so a match can only ever narrow it."""
        long_segment = [segment(0, 0.0, 40.0, ' '.join(f'word{i}' for i in range(40)))]
        answerer = replies(reply('yes', 0, ' '.join(f'word{i}' for i in range(40))))
        [(_, span)] = answer_conversation(long_segment, ['q?'], answerer)

        assert span[0] >= 0.0 and span[1] <= 40.0

    def test_past_the_deadline_questions_are_guessed_not_asked(self, segments):
        """Ten slow calls must not run past the budget: silence ends the attempt."""
        import answering

        asked = []

        def slow(prompt, timeout=None):
            asked.append(prompt)
            return reply('no')

        original = answering.DEADLINE_SECONDS
        answering.DEADLINE_SECONDS = -1.0  # every question is already late
        try:
            results = answer_conversation(segments, ['a?', 'b?', 'c?'], slow)
        finally:
            answering.DEADLINE_SECONDS = original

        assert asked == []
        assert all(answer is True and span is not None for answer, span in results)

    def test_each_call_is_bounded_by_what_is_left_of_the_budget(self, segments):
        """A deadline checked only between calls cannot stop one slow call.

        The check decides whether to ask; the timeout decides how long we wait.
        Without the second, a call starting just inside the deadline returns
        long after it, and the request misses the budget it was protecting.
        """
        import time

        seen = []

        def recording(prompt, timeout=None):
            seen.append(timeout)
            return reply('no')

        deadline = time.perf_counter() + 5.0
        answer_conversation(segments, ['a?', 'b?', 'c?'], recording, deadline=deadline)

        assert len(seen) == 3
        assert all(t is not None and 0 < t <= 5.0 for t in seen)
        # Each call is handed less than the one before it.
        assert seen == sorted(seen, reverse=True)

    def test_yes_never_returns_a_null_span(self, segments):
        """A wrong span scores the same as none; null is guaranteed zero."""
        answerer = replies(reply('yes', None, None))
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span is not None


class TestMalformedReplies:
    def test_bare_no_without_json_is_understood(self, segments):
        answerer = replies('No, the conversation says 100 mg, not 200 mg.')
        [(answer, span)] = answer_conversation(segments, ['Is the dose 200 mg?'], answerer)

        assert answer is False
        assert span is None

    def test_bare_yes_without_json_is_understood(self, segments):
        answerer = replies('Yes.')
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span is not None

    def test_unparseable_reply_guesses_yes_with_a_span(self, segments):
        answerer = replies('\x00\x00 ???')
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span is not None

    def test_segment_index_wrapped_in_a_list_is_understood(self, segments):
        """qwen3 copies the [n] notation out of the prompt and returns [3]."""
        answerer = replies(json.dumps(
            {'answer': 'yes', 'segment': [3], 'quote': 'one hundred milligrams'}))
        [(_, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert 20.0 < span[0] and span[1] < 26.0

    def test_segment_index_as_a_bracketed_string_is_understood(self, segments):
        answerer = replies(json.dumps(
            {'answer': 'yes', 'segment': '[3]', 'quote': 'one hundred milligrams'}))
        [(_, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert 20.0 < span[0] and span[1] < 26.0

    def test_out_of_range_segment_index_still_returns_a_span(self, segments):
        answerer = replies(reply('yes', 99, 'one hundred milligrams'))
        [(answer, span)] = answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        assert answer is True
        assert span is not None

    def test_a_raising_answerer_costs_one_question_not_the_conversation(self, segments):
        def answerer(prompt, timeout=None):
            raise RuntimeError('ollama fell over')

        results = answer_conversation(segments, ['a?', 'b?', 'c?'], answerer)

        assert len(results) == 3
        assert all(answer is True and span is not None for answer, span in results)


class TestSharedPrefix:
    """ADR-0001. Breaking this silently costs ~70 s per conversation."""

    def test_every_prompt_opens_with_a_byte_identical_transcript_prefix(self, segments):
        answerer = replies(*[reply('yes', 0, 'Good morning') for _ in range(10)])
        questions = [f'question {i}?' for i in range(10)]

        answer_conversation(segments, questions, answerer)

        header = build_transcript_header(segments)
        assert len(answerer.prompts) == 10
        assert all(prompt.startswith(header) for prompt in answerer.prompts)

    def test_the_question_appears_after_the_transcript(self, segments):
        answerer = replies(reply('no'))
        answer_conversation(segments, ['Is the dose 100 mg?'], answerer)

        prompt = answerer.prompts[0]
        assert prompt.index('Is the dose 100 mg?') > prompt.index('Doctor Norgaard')

    def test_the_header_carries_segment_indices_and_timestamps(self, segments):
        header = build_transcript_header(segments)

        assert '[3]' in header
        assert '20.0' in header
        assert 'Doctor Norgaard' in header


@pytest.fixture
def punctuated():
    """Segments whose sentences run across the segment boundary.

    Segment 0 ends mid-sentence and segment 1 finishes it, which is the ordinary
    case: the ASR divides on silence, not on syntax.
    """
    return [
        segment(0, 0.0, 4.0, 'Good morning. Your chest and heart'),
        segment(1, 4.0, 8.0, 'both sound normal. Nothing abnormal to report.'),
        segment(2, 8.0, 14.0, 'Sporinox. One hundred milligrams daily for two weeks. '
                              'Take it after a meal.'),
    ]


class TestPassages:
    """The span we return is a whole number of sentences, not a quote or a segment."""

    def test_a_quote_is_widened_to_the_sentence_containing_it(self, punctuated):
        answerer = replies(reply('yes', 2, 'after a meal'))
        [(_, span)] = answer_conversation(punctuated, ['Is it taken after a meal?'], answerer)

        # 'Take it after a meal.' starts before the quoted words do.
        sentence_start = punctuated[2]['words'][-5]['start']
        assert span == inset_of((sentence_start, 14.0))

    def test_a_sentence_split_across_two_segments_is_returned_whole(self, punctuated):
        """The point of the change: the segment boundary must not truncate it."""
        answerer = replies(reply('yes', 1, 'both sound normal'))
        [(_, span)] = answer_conversation(
            punctuated, ['Is the heart normal?'], answerer)

        # 'Your chest and heart both sound normal.' begins in segment 0.
        assert span[0] < 4.0 < span[1]

    def test_the_sentence_carrying_the_question_terms_is_chosen(self, punctuated):
        """A quote spanning several sentences is narrowed, not just widened."""
        answerer = replies(reply(
            'yes', 2, 'Sporinox. One hundred milligrams daily for two weeks. '
                      'Take it after a meal.'))
        [(_, span)] = answer_conversation(
            punctuated, ['Is the daily dose one hundred milligrams?'], answerer)

        assert span[1] - span[0] < 6.0
        assert span[0] >= punctuated[2]['words'][1]['start']

    def test_a_passage_reaches_past_the_quote_to_the_question_it_answers(self, punctuated):
        """An annotated passage is the exchange, not the half the model quoted.

        The quote lies wholly inside the closing sentence; the question's own
        terms are in the sentence before it, within the reach the span logic
        allows, so the passage returned covers both.
        """
        answerer = replies(reply('yes', 2, 'Take it after a meal.'))
        [(_, span)] = answer_conversation(
            punctuated, ['Is the two week course taken after a meal?'], answerer)

        # 'One hundred milligrams daily for two weeks.' precedes the quote.
        # Both edges sit within the constant inset of the sentence bounds: what
        # is asserted here is which sentences the passage covers, not the exact
        # floats, and the inset moves each edge by |PAD_SECONDS|.
        inset = abs(answering.PAD_SECONDS)
        dose_sentence = punctuated[2]['words'][1]['start']
        assert span[0] <= dose_sentence + inset
        assert span[1] == pytest.approx(14.0 - inset, abs=0.01)

    def test_reaching_outward_does_not_overrule_the_sentence_quoted(self, punctuated):
        """The reach breaks ties towards neighbours, it does not abandon the quote."""
        answerer = replies(reply('yes', 0, 'Good morning'))
        [(_, span)] = answer_conversation(punctuated, ['Did the doctor say good morning?'], answerer)

        # The opening sentence, give or take the constant inset.
        assert span[0] <= abs(answering.PAD_SECONDS) + 0.01

    def test_the_quote_is_still_matched_only_inside_the_named_segment(self, punctuated):
        """Widening must not reintroduce conversation-wide matching."""
        answerer = replies(reply('yes', 0, 'after a meal'))
        [(_, span)] = answer_conversation(
            punctuated, ['Is it taken after a meal?'], answerer)

        # Those words are only in segment 2, so segment 0 cannot match them and
        # the span must stay where the model pointed.
        assert span[0] < 8.0

    def test_an_unpunctuated_transcript_does_not_widen_to_the_whole_conversation(self):
        """No punctuation means no sentences; a 40 s 'sentence' is not one."""
        unpunctuated = [
            segment(0, 0.0, 20.0, ' '.join(f'word{i}' for i in range(20))),
            segment(1, 20.0, 40.0, ' '.join(f'other{i}' for i in range(20))),
        ]
        answerer = replies(reply('yes', 1, 'other3 other4 other5'))
        [(_, span)] = answer_conversation(unpunctuated, ['q?'], answerer)

        assert span[1] - span[0] < 10.0

    def test_a_yes_still_never_returns_a_null_span(self, punctuated):
        answerer = replies(reply('yes', None, None))
        [(answer, span)] = answer_conversation(punctuated, ['q?'], answerer)

        assert answer is True
        assert span is not None


class TestTheSecondPass:
    """It may improve a span. It may never cost one.

    The second pass asks the model to choose among the candidate passages the
    heuristic already built. Every failure -- a refusal, an unreadable reply, an
    out-of-range number, no time left -- must land on the span we already had.
    """

    def test_it_is_off_by_default(self):
        assert answering.SECOND_PASS is False

    def test_an_unreadable_choice_keeps_the_heuristic_span(self, punctuated):
        chosen = self._with_second_pass(punctuated, 'I could not say')
        assert chosen == self._without(punctuated)

    def test_a_number_out_of_range_keeps_the_heuristic_span(self, punctuated):
        chosen = self._with_second_pass(punctuated, '99')
        assert chosen == self._without(punctuated)

    def test_a_failing_second_call_keeps_the_heuristic_span(self, punctuated):
        def explode(prompt, timeout=None):
            if 'Which passage' in prompt:
                raise RuntimeError('model down')
            return reply('yes', 2, 'after a meal')

        original = answering.SECOND_PASS
        answering.SECOND_PASS = True
        try:
            [(_, span)] = answer_conversation(
                punctuated, ['Is it taken after a meal?'], explode)
        finally:
            answering.SECOND_PASS = original
        assert span == self._without(punctuated)

    def test_a_confident_ranking_is_never_escalated(self, punctuated):
        """No call is made when the heuristic already has a clear winner."""
        asked = []

        def counting(prompt, timeout=None):
            asked.append(prompt)
            return reply('yes', 2, 'after a meal')

        original_margin = answering.SELECT_MARGIN
        original = answering.SECOND_PASS
        answering.SECOND_PASS = True
        answering.SELECT_MARGIN = 0.0   # nothing is ever close enough
        try:
            answer_conversation(punctuated, ['Is it taken after a meal?'], counting)
        finally:
            answering.SECOND_PASS = original
            answering.SELECT_MARGIN = original_margin

        assert len(asked) == 1, 'the second pass should not have been called'

    def test_the_selection_prompt_still_opens_with_the_shared_header(self, punctuated):
        """ADR-0001: break this and a conversation costs ~70 s with no signal."""
        header = build_transcript_header(punctuated)
        candidates = [answering.Candidate(1.0, (0.0, 1.0), 'first'),
                      answering.Candidate(0.9, (2.0, 3.0), 'second')]
        prompt = answering.build_selection_prompt(header, 'Why?', candidates)

        assert prompt.startswith(header)
        assert '[1] first' in prompt and '[2] second' in prompt

    def test_parse_selection_reads_a_number_and_rejects_the_rest(self):
        assert answering.parse_selection('2', 5) == 1
        assert answering.parse_selection('The answer is [3].', 5) == 2
        assert answering.parse_selection('9', 5) is None
        assert answering.parse_selection('none of them', 5) is None
        assert answering.parse_selection('', 5) is None

    # --- helpers ---

    def _without(self, punctuated):
        answerer = replies(reply('yes', 2, 'after a meal'))
        [(_, span)] = answer_conversation(
            punctuated, ['Is it taken after a meal?'], answerer)
        return span

    def _with_second_pass(self, punctuated, selection):
        def answerer(prompt, timeout=None):
            if 'Which passage' in prompt:
                return selection
            return reply('yes', 2, 'after a meal')

        original = answering.SECOND_PASS
        answering.SECOND_PASS = True
        try:
            [(_, span)] = answer_conversation(
                punctuated, ['Is it taken after a meal?'], answerer)
        finally:
            answering.SECOND_PASS = original
        return span
