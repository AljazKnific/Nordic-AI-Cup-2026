"""Score an answering strategy against the supplied questions. Dev only.

Runs off cached transcripts, so there is no server, no audio decoding and no ASR
in the loop — one pass takes about a minute instead of about fifteen.

    python tools/score.py --ceiling    # no LLM: what segment bounds alone are worth
    python tools/score.py              # the real thing, through Ollama
    python tools/score.py --limit 5    # first five conversations, while iterating

``--ceiling`` is the honest baseline for evidence work: for every annotated span
it takes the best-overlapping segment, which is the most that pointing at whole
segments could ever score. Anything the quote matcher does has to beat it.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcripts  # noqa: E402
from answering import answer_conversation, segment_bounds  # noqa: E402
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)


def best_segment_span(segments: List[Dict], gold: Span) -> Span:
    """The span an oracle would return if it could only point at whole segments."""
    return segment_bounds(max(segments, key=lambda s: temporal_iou(gold, segment_bounds(s))))


def best_word_span(segments, gold):
    """The best span any word-level trimming could produce.

    The upper bound on the quote matcher: the contiguous run of words, anywhere
    in the conversation, with the highest tIoU against the annotated passage.
    """
    words = [w for segment in segments for w in (segment.get('words') or [])]
    if not words:
        return None

    best, best_iou = None, -1.0
    for i in range(len(words)):
        # Gold passages run a median of 2.9 s, so a run longer than ~30 words
        # cannot be the answer; capping keeps this quadratic scan cheap.
        for j in range(i, min(i + 30, len(words))):
            span = (float(words[i]['start']), float(words[j]['end']))
            iou = temporal_iou(gold, span)
            if iou > best_iou:
                best, best_iou = span, iou
    return best


def report(rows: List[Dict], answers: List[bool], spans: List[Optional[Span]]) -> None:
    by_type: Dict[str, List[bool]] = {}
    ious: List[float] = []

    for row, answer, span in zip(rows, answers, spans):
        correct = (answer is True) == (row['label'] == '1')
        by_type.setdefault(row['question_type'], []).append(correct)

        gold = gold_evidence(row)
        if row['label'] == '1' and gold is not None:
            ious.append(temporal_iou(gold, span if answer else None))

    correct_all = [c for values in by_type.values() for c in values]
    accuracy = sum(correct_all) / len(correct_all) if correct_all else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    print(f'\n{"":<16}{"correct":>10}{"n":>7}')
    for question_type in ('positive', 'hard_negative', 'off_topic'):
        values = by_type.get(question_type)
        if values:
            print(f'{question_type:<16}{sum(values) / len(values):>10.3f}{len(values):>7}')

    print(f'\naccuracy   {accuracy:.3f}')
    print(f'mean tIoU  {mean_iou:.3f}  (over {len(ious)} annotated yes questions)')
    print(f'SCORE      {0.4 * accuracy + 0.6 * mean_iou:.3f}   = 0.4 x accuracy + 0.6 x tIoU')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceiling', action='store_true',
                        help='no LLM: oracle answers, best-overlapping segment as the span')
    parser.add_argument('--ceiling-words', action='store_true',
                        help='no LLM: oracle answers, best word-level run as the span')
    parser.add_argument('--limit', type=int, default=None,
                        help='only the first N conversations')
    parser.add_argument('--verbose', action='store_true', help='per-call LLM timings')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(message)s')

    groups = group_questions_by_conversation()
    if args.limit:
        groups = groups[:args.limit]

    oracle = args.ceiling or args.ceiling_words

    answerer = None
    if not oracle:
        import ollama_client
        answerer = ollama_client.answer
        ollama_client.warm_up()

    all_rows: List[Dict] = []
    all_answers: List[bool] = []
    all_spans: List[Optional[Span]] = []
    missing = 0
    started = time.perf_counter()

    for audio_filename, rows in groups:
        segments = transcripts.load(audio_filename)
        if segments is None:
            missing += 1
            continue

        if oracle:
            # Perfect answers; the span is whatever the chosen granularity
            # could at best produce. The most that approach could ever score.
            locate = best_word_span if args.ceiling_words else best_segment_span
            answers = [row['label'] == '1' for row in rows]
            spans = [
                locate(segments, gold_evidence(row))
                if row['label'] == '1' and gold_evidence(row) else None
                for row in rows
            ]
        else:
            questions = [row['question'] for row in rows]
            results = answer_conversation(segments, questions, answerer)
            answers = [answer for answer, _ in results]
            spans = [span for _, span in results]
            print(f'{audio_filename}  {time.perf_counter() - started:6.1f} s', flush=True)

        all_rows.extend(rows)
        all_answers.extend(answers)
        all_spans.extend(spans)

    if missing:
        print(f'{missing} conversations had no cached transcript '
              f'(run tools/build_transcripts.py)')
    if not all_rows:
        print('Nothing to score.')
        return 1

    print(f'\n{len(all_rows)} questions over '
          f'{len(all_rows) // 10} conversations in {time.perf_counter() - started:.1f} s')
    report(all_rows, all_answers, all_spans)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
