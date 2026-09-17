"""What is actually wrong, rather than a single score. Dev only. (Ticket 5.)

Three readouts, all off the captured replies in ``diagnostics/replies.json``
(see ``tools/capture_replies.py``), so this costs a second rather than twelve
minutes:

1. accuracy by question type, naming every question that was answered wrongly
2. how often each fallback layer fires, counted by capturing the ``FALLBACK``
   log records the real code path emits
3. how often an entity a question names -- a drug, a dose, a number -- is
   missing from the transcript that was supposed to contain it

The third is the input to ticket 6. Only ``positive`` questions are counted
there: a hard negative names a decoy term that was never spoken, so its term
being absent from the transcript is the transcript being right.
"""

import argparse
import difflib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import answering  # noqa: E402
import transcripts  # noqa: E402
from answering import (  # noqa: E402
    _find_segment, parse_reply, resolve_span, sentences_of,
)
from utils import group_questions_by_conversation  # noqa: E402

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'

# A term is judged present if the transcript spells it exactly; "close" if some
# transcript word is nearly it, which is what a garbled drug name looks like.
NEAR = 0.75

_NUMBER = re.compile(r'\b\d+(?:[.,]\d+)?\b')
_CAPITALISED = re.compile(r'\b[A-Z][a-z]{3,}\b')


class Fallbacks(logging.Handler):
    """Counts the FALLBACK layers by the fixed text each one logs."""

    LAYERS = (
        'quote-unmatched',
        'reply was not JSON',
        'reply unreadable',
        'no usable segment index',
        'sentence longer than',
        'past the',
    )

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.counts = Counter()

    def emit(self, record):
        message = record.getMessage()
        if not message.startswith('FALLBACK'):
            return
        for layer in self.LAYERS:
            if layer in message:
                self.counts[layer] += 1
                return
        self.counts['other'] += 1


def question_terms(question: str):
    """The entities a question names: drug-like words and numbers.

    A capitalised word that is not the first word of the question is the usable
    signal for a drug name -- no lexicon, because the evaluation drugs are
    unseen and one built from these 39 conversations would not contain them.
    """
    body = question.split(' ', 1)[-1]
    return sorted(set(_CAPITALISED.findall(body)) | set(_NUMBER.findall(question)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replies', default=str(REPLIES))
    args = parser.parse_args()

    replies = json.loads(Path(args.replies).read_text())

    fallbacks = Fallbacks()
    answering.logger.addHandler(fallbacks)
    answering.logger.setLevel(logging.INFO)

    by_type = defaultdict(list)
    wrong = []
    missing = []
    checked_terms = 0
    conversations = 0

    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        conversations += 1
        sentences = sentences_of(segments)
        spoken = {
            word.casefold().strip('.,?!;:')
            for segment in segments for word in segment['text'].split()
        }

        for row in rows:
            text = replies.get(row['question_id'])
            if text is None:
                continue

            answer, index, quote = parse_reply(text)
            if answer is not False:
                segment = _find_segment(segments, index)
                if segment is None:
                    segment = answering._best_keyword_segment(segments, row['question'])
                resolve_span(segment, quote, sentences, row['question'])

            said_yes = answer is not False
            correct = said_yes == (row['label'] == '1')
            by_type[row['question_type']].append(correct)
            if not correct:
                wrong.append((row, 'yes' if said_yes else 'no'))

            # Entity check: only where the term must have been spoken.
            if row['question_type'] != 'positive':
                continue
            for term in question_terms(row['question']):
                checked_terms += 1
                if term.casefold() in spoken:
                    continue
                close = difflib.get_close_matches(term.casefold(), spoken, 1, NEAR)
                missing.append((row['question_id'], term, close[0] if close else None))

    print(f'{conversations} conversations, {sum(len(v) for v in by_type.values())} questions\n')

    print(f'{"question type":<18}{"correct":>9}{"n":>6}')
    for question_type in ('positive', 'hard_negative', 'off_topic'):
        values = by_type.get(question_type)
        if values:
            print(f'{question_type:<18}{sum(values) / len(values):>9.3f}{len(values):>6}')
    total = [c for v in by_type.values() for c in v]
    print(f'{"all":<18}{sum(total) / len(total):>9.3f}{len(total):>6}')

    print(f'\nwrong answers ({len(wrong)}):')
    for row, said in wrong:
        print(f'  {row["question_id"]:<26} {row["question_type"]:<14} '
              f'said {said}, gold {"yes" if row["label"] == "1" else "no"}')
        print(f'    {row["question"]}')

    print(f'\nfallback layers fired (over {conversations} conversations):')
    if not fallbacks.counts:
        print('  none')
    for layer, count in fallbacks.counts.most_common():
        print(f'  {layer:<26}{count:>5}')
    print('  (the deadline and answerer-exception layers cannot fire in a replay;'
          '\n   they are counted from a live tools/score.py run instead)')

    print(f'\nentities named by positive questions: {checked_terms} checked, '
          f'{len(missing)} not spelled the same in the transcript')
    garbled = [m for m in missing if m[2]]
    print(f'  {len(garbled)} have a near-match in the transcript (probable ASR error)')
    print(f'  {len(missing) - len(garbled)} have no near-match '
          f'(paraphrase, or a number written differently)')
    for question_id, term, close in missing:
        print(f'    {question_id:<26} {term:<18} -> {close or "(nothing close)"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
