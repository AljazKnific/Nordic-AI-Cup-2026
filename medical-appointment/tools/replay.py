"""Replay captured model replies through the span logic. Dev only.

The span code is a pure function of (segments, reply); the model is not in the
loop. ``tools/capture_replies.py`` records one raw reply per question once
(~12 minutes), and this scores any change to parsing or span resolution against
them in under a second. Use it to iterate; use ``tools/score.py`` to confirm.

``--diagnose`` attributes every lost point on an annotated yes question, which
is the thing ticket 4 asks for. It compares what we returned against the best
run of sentences reachable from the segment the model named:

* ``best-available``    we returned the best that segment allowed; nothing left
* ``wrong-sentences``   the passage was reachable, we chose the wrong sentences
* ``wrong-mention``     the named segment holds no sentence near the passage --
  either the wrong segment, or the right fact stated in a different place
* ``answered-no``       a positive answered no, which scores a tIoU of 0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcripts  # noqa: E402
import answering  # noqa: E402
from answering import (  # noqa: E402
    _find_segment, parse_reply, resolve_span, sentences_of,
)
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'


# Passages run to at most a few sentences; four is well past the longest.
SENTENCE_RUN = 4


def best_sentence_run(sentences, gold: Span) -> float:
    """The most any whole-sentence span over these sentences could score."""
    best = 0.0
    for i in range(len(sentences)):
        for j in range(i, min(i + SENTENCE_RUN, len(sentences))):
            best = max(best, temporal_iou(gold, (sentences[i].start, sentences[j].end)))
    return best


def sentences_in(sentences, segment) -> list:
    """The sentences the named segment reaches -- what its choice made available."""
    return [
        s for s in sentences
        if min(float(segment['end']), s.end) - max(float(segment['start']), s.start) > 0.05
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replies', default=str(REPLIES))
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--worst', type=int, default=0, help='show the N worst questions')
    args = parser.parse_args()

    replies: Dict[str, str] = json.loads(Path(args.replies).read_text())

    rows_scored = 0
    correct = 0
    ious: List[float] = []
    causes: Dict[str, List[float]] = {}
    detail = []

    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = sentences_of(segments)
        for row in rows:
            text = replies.get(row['question_id'])
            if text is None:
                continue
            rows_scored += 1

            answer, index, quote = parse_reply(text)
            if answer is False:
                span: Optional[Span] = None
                predicted_yes = False
            else:
                predicted_yes = True
                segment = _find_segment(segments, index)
                if segment is None:
                    segment = answering._best_keyword_segment(segments, row['question'])
                span = resolve_span(segment, quote, sentences, row["question"])

            correct += (predicted_yes == (row['label'] == '1'))

            gold = gold_evidence(row)
            if row['label'] != '1' or gold is None:
                continue
            iou = temporal_iou(gold, span if predicted_yes else None)
            ious.append(iou)

            if not args.diagnose and not args.worst:
                continue

            # Attribution: against the best this segment choice allowed.
            named = _find_segment(segments, index) if predicted_yes else None
            if not predicted_yes:
                cause, headroom = 'answered-no', best_sentence_run(sentences, gold)
            else:
                if named is None:
                    named = answering._best_keyword_segment(segments, row['question'])
                reachable = sentences_in(sentences, named)
                allowed = best_sentence_run(reachable, gold) if reachable else 0.0
                if allowed < 0.45:
                    cause = 'wrong-mention'
                    headroom = best_sentence_run(sentences, gold)
                elif iou >= allowed - 0.03:
                    cause, headroom = 'best-available', iou
                else:
                    cause, headroom = 'wrong-sentences', allowed
            causes.setdefault(cause, []).append(max(0.0, headroom - iou))
            detail.append((iou, headroom, cause, row, index, quote, span, gold, named))

    n = len(ious)
    accuracy = correct / rows_scored if rows_scored else 0.0
    mean_iou = sum(ious) / n if n else 0.0
    print(f'{rows_scored} questions replayed')
    print(f'accuracy   {accuracy:.3f}')
    print(f'mean tIoU  {mean_iou:.3f}  (over {n} annotated yes questions)')
    print(f'SCORE      {0.4 * accuracy + 0.6 * mean_iou:.3f}')

    if args.diagnose:
        print(f'\n{"cause":<20}{"n":>5}{"lost per q":>13}{"recoverable":>14}')
        for cause, lost in sorted(causes.items(), key=lambda kv: -sum(kv[1])):
            print(f'{cause:<20}{len(lost):>5}{sum(lost) / len(lost):>13.3f}'
                  f'{sum(lost) / n:>14.3f}')

    if args.worst:
        detail.sort(key=lambda d: d[0])
        for iou, head, cause, row, index, quote, span, gold, named in detail[:args.worst]:
            print(f'\n--- {row["question_id"]}  iou={iou:.2f} headroom={head:.2f} [{cause}]')
            print(f'  Q: {row["question"]}')
            print(f'  gold {gold}  pred {span}  named segment {index}')
            if named is not None:
                print(f'  seg[{named["index"]}] {named["start"]}-{named["end"]}: {named["text"]}')
            print(f'  quote: {quote!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
