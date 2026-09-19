"""Fit the extent heuristic's constants against the annotated passages. Dev only.

The span code is a pure function of (located span, sentences, question) and six
module constants in :mod:`answering`. This searches those constants against the
frozen replies -- no model in the loop, so a whole search costs seconds.

**The protocol matters more than the search.** Six free parameters against 195
annotated passages from 39 conversations will always improve the number they
were fitted on. So the conversations are split, the search only ever sees the
fit half, and the decision is made on the half it never saw:

    fit on 25 conversations -> report on the held-out 14

A gain on the fit set that does not survive the holdout is overfitting, and the
answer is no. The bootstrap resamples *conversations*, not questions: ten
questions about one consultation share a transcript and are not independent, so
resampling questions would report an interval several times too narrow.

    python tools/fit.py                 # search, then report on the holdout
    python tools/fit.py --seed 7        # a different split
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import answering  # noqa: E402
import transcripts  # noqa: E402
from answering import _find_segment, parse_reply, resolve_span, sentences_of  # noqa: E402
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'

# Each parameter, with the grid the search may move it over. Ranges bracket the
# shipped value generously in both directions: a search that can only confirm
# the current setting is not a search.
GRIDS: Dict[str, List] = {
    'REACH_SECONDS': [round(0.25 * i, 2) for i in range(0, 13)],        # 0 - 3.0
    'ANCHOR_WEIGHT': [round(0.1 * i, 2) for i in range(0, 21)],         # 0 - 2.0
    'LENGTH_PENALTY': [round(0.05 * i, 2) for i in range(0, 21)],       # 0 - 1.0
    'TARGET_SECONDS': [round(1.0 + 0.25 * i, 2) for i in range(0, 21)],  # 1.0 - 6.0
    'MAX_RUN_SENTENCES': [1, 2, 3, 4, 5, 6],
    'PAD_SECONDS': [round(0.05 * i, 2) for i in range(0, 21)],          # 0 - 1.0
}

PASSES = 4


class Item(NamedTuple):
    conversation: str
    located: Optional[Span]   # None when the model answered no: always scores 0
    sentences: tuple
    question: str
    gold: Span


def load_items(replies_path: Path) -> List[Item]:
    """One entry per annotated yes question, with the model's reply resolved.

    The located span is captured *before* rounding out, by resolving the quote
    against no sentences at all -- which is what :func:`answering.to_passage`
    returns untouched. Everything the search varies happens after this point.
    """
    for name in GRIDS:
        setattr(answering, name, 0 if name == 'PAD_SECONDS' else getattr(answering, name))
    answering.PAD_SECONDS = 0.0

    replies: Dict[str, str] = json.loads(replies_path.read_text())
    items: List[Item] = []
    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = tuple(sentences_of(segments))
        for row in rows:
            text = replies.get(row['question_id'])
            gold = gold_evidence(row)
            if text is None or row['label'] != '1' or gold is None:
                continue
            answer, index, quote = parse_reply(text)
            if answer is False:
                items.append(Item(audio_filename, None, sentences, row['question'], gold))
                continue
            segment = _find_segment(segments, index)
            if segment is None:
                segment = answering._best_keyword_segment(segments, row['question'])
            located = resolve_span(segment, quote, (), row['question'])
            items.append(Item(audio_filename, located, sentences, row['question'], gold))
    return items


def apply(params: Dict) -> None:
    for name, value in params.items():
        setattr(answering, name, value)


def mean_iou(items: List[Item], params: Dict) -> float:
    apply(params)
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        if item.located is None:
            continue          # answered no: tIoU 0, and no parameter changes that
        passage = answering.to_passage(item.located, item.sentences, item.question)
        total += temporal_iou(item.gold, passage)
    return total / len(items)


def search(items: List[Item], start: Dict) -> Dict:
    """Coordinate descent over the grids. Only ever shown the fit set."""
    best = dict(start)
    best_score = mean_iou(items, best)
    for _ in range(PASSES):
        improved = False
        for name, grid in GRIDS.items():
            for value in grid:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: value})
                score = mean_iou(items, trial)
                if score > best_score + 1e-9:
                    best, best_score = trial, score
                    improved = True
        if not improved:
            break
    return best


def bootstrap_delta(
    items: List[Item], base: Dict, fitted: Dict, rounds: int, rng: random.Random,
) -> Tuple[float, float]:
    """A 95% interval on the change, resampling whole conversations."""
    by_conversation: Dict[str, List[Item]] = {}
    for item in items:
        by_conversation.setdefault(item.conversation, []).append(item)
    names = list(by_conversation)

    deltas = []
    for _ in range(rounds):
        drawn = [rng.choice(names) for _ in names]
        sample = [i for name in drawn for i in by_conversation[name]]
        deltas.append(mean_iou(sample, fitted) - mean_iou(sample, base))
    deltas.sort()
    return deltas[int(0.025 * rounds)], deltas[int(0.975 * rounds)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replies', default=str(REPLIES))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--holdout', type=int, default=14)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()

    base = {name: getattr(answering, name) for name in GRIDS}
    items = load_items(Path(args.replies))

    rng = random.Random(args.seed)
    conversations = sorted({item.conversation for item in items})
    rng.shuffle(conversations)
    held = set(conversations[:args.holdout])
    fit_items = [i for i in items if i.conversation not in held]
    held_items = [i for i in items if i.conversation in held]

    print(f'{len(items)} annotated passages over {len(conversations)} conversations')
    print(f'fit on {len(conversations) - len(held)} ({len(fit_items)} passages), '
          f'held out {len(held)} ({len(held_items)} passages), seed {args.seed}\n')

    fitted = search(fit_items, base)

    rows = [
        ('fit set', mean_iou(fit_items, base), mean_iou(fit_items, fitted)),
        ('HELD OUT', mean_iou(held_items, base), mean_iou(held_items, fitted)),
        ('all', mean_iou(items, base), mean_iou(items, fitted)),
    ]
    print(f'{"":<10}{"shipped":>10}{"fitted":>10}{"delta":>10}')
    for label, before, after in rows:
        print(f'{label:<10}{before:>10.4f}{after:>10.4f}{after - before:>+10.4f}')

    low, high = bootstrap_delta(held_items, base, fitted, args.bootstrap, random.Random(args.seed + 1))
    print(f'\nheld-out delta 95% CI (resampling conversations): {low:+.4f} to {high:+.4f}')

    print('\nparameters:')
    for name in GRIDS:
        mark = '' if base[name] == fitted[name] else '   <-- changed'
        print(f'  {name:<20}{str(base[name]):>8} -> {str(fitted[name]):>8}{mark}')

    verdict = 'SHIP' if rows[1][2] > rows[1][1] and low > 0 else 'DO NOT SHIP'
    print(f'\n{verdict}: held-out {"improved" if rows[1][2] > rows[1][1] else "did not improve"}, '
          f'interval {"excludes" if low > 0 else "includes"} zero')
    apply(base)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
