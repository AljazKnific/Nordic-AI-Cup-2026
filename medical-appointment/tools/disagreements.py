"""Where the model's own choice disagrees with the annotation, and how. Dev only.

`tools/replay.py --diagnose` says *how much* tIoU each cause costs.  This asks
the next question: **in which direction, and is it the same direction every
time?** A loss that is systematic can be argued with -- in the prompt, or in the
span logic.  A loss that is symmetric noise cannot, and knowing which is which
is the difference between an experiment worth running and one already answered.

Everything here is measured against the 39 supplied conversations, which are the
only ones with annotations.  The hosted conversations never come with labels, so
they can never become examples; what they *can* do is show whether the model
behaves the same way on them, which `--requests` checks without needing any.

    python tools/disagreements.py                      # the report
    python tools/disagreements.py --worst 10           # the costliest cases
    python tools/disagreements.py --requests diagnostics/requests
"""

import argparse
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import answering  # noqa: E402
import transcripts  # noqa: E402
from answering import (  # noqa: E402
    _find_segment, _normalise, parse_reply, resolve_span, sentences_of,
)
from utils import (  # noqa: E402
    gold_evidence, group_questions_by_conversation, temporal_iou,
)

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'


def summarise(values: List[float], unit: str = 's') -> str:
    if not values:
        return 'none'
    values = sorted(values)
    return (f'median {stats.median(values):+.2f}{unit}  '
            f'p10 {values[len(values) // 10]:+.2f}{unit}  '
            f'p90 {values[min(len(values) - 1, 9 * len(values) // 10)]:+.2f}{unit}  '
            f'n={len(values)}')


def sign_split(values: List[float], label_low: str, label_high: str) -> str:
    """How lopsided a signed error is -- the whole point of the report."""
    if not values:
        return 'none'
    low = sum(1 for v in values if v < -0.05)
    high = sum(1 for v in values if v > 0.05)
    same = len(values) - low - high
    return (f'{label_low} {low}  |  within 0.05s {same}  |  {label_high} {high}'
            f'   ({100 * max(low, high) / len(values):.0f}% lopsided)')


def collect(replies_path: Path) -> List[Dict]:
    replies: Dict[str, str] = json.loads(replies_path.read_text())
    found: List[Dict] = []

    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = sentences_of(segments)
        for row in rows:
            text = replies.get(row['question_id'])
            gold = gold_evidence(row)
            if text is None or row['label'] != '1' or gold is None:
                continue

            answer, index, quote = parse_reply(text)
            if answer is False:
                found.append({'row': row, 'gold': gold, 'cause': 'answered-no',
                              'iou': 0.0, 'quote': None, 'named': None})
                continue

            named = _find_segment(segments, index)
            if named is None:
                named = answering._best_keyword_segment(segments, row['question'])
            span = resolve_span(named, quote, sentences, row['question'])
            iou = temporal_iou(gold, span)

            # Does the quote's own text turn up inside the gold passage?
            in_gold = [s for s in sentences
                       if min(gold[1], s.end) - max(gold[0], s.start) > 0.05]
            gold_words = _normalise(' '.join(s.text for s in in_gold)).split()
            quote_words = _normalise(quote or '').split()
            overlap = (len(set(quote_words) & set(gold_words)) / len(quote_words)
                       if quote_words else 0.0)

            found.append({
                'row': row, 'gold': gold, 'span': span, 'iou': iou,
                'quote': quote, 'named': named,
                'start_error': span[0] - gold[0],
                'end_error': span[1] - gold[1],
                'length_error': (span[1] - span[0]) - (gold[1] - gold[0]),
                'quote_in_gold': overlap,
                'segment_error': (float(named['start']) - gold[0]) if named else None,
                'cause': None,
            })

    for item in found:
        if item['cause'] == 'answered-no':
            continue
        if item['quote_in_gold'] < 0.35:
            item['cause'] = 'wrong-mention'
        elif item['iou'] >= 0.75:
            item['cause'] = 'close-enough'
        else:
            item['cause'] = 'wrong-extent'
    return found


def report(found: List[Dict], worst: int) -> None:
    scored = [f for f in found if f['cause'] != 'answered-no']
    print(f'{len(found)} annotated passages, {len(scored)} answered yes\n')

    print(f'{"cause":<16}{"n":>5}{"mean tIoU":>12}{"tIoU lost":>12}')
    for cause in ('close-enough', 'wrong-extent', 'wrong-mention', 'answered-no'):
        group = [f for f in found if f['cause'] == cause]
        if not group:
            continue
        mean = sum(f['iou'] for f in group) / len(group)
        print(f'{cause:<16}{len(group):>5}{mean:>12.3f}'
              f'{sum(1 - f["iou"] for f in group) / len(found):>12.3f}')

    print('\n--- Is the error lopsided? (negative = we are early/short) ---')
    for name, key, low, high in (
        ('span start', 'start_error', 'we start EARLY', 'we start LATE'),
        ('span end', 'end_error', 'we end EARLY', 'we end LATE'),
        ('span length', 'length_error', 'we are SHORT', 'we are LONG'),
    ):
        values = [f[key] for f in scored]
        print(f'  {name:<12} {summarise(values)}')
        print(f'  {"":<12} {sign_split(values, low, high)}')

    print('\n--- Where the quote lands, when the mention is wrong ---')
    missed = [f for f in scored if f['cause'] == 'wrong-mention'
              and f['segment_error'] is not None]
    if missed:
        values = [f['segment_error'] for f in missed]
        print(f'  named segment vs gold  {summarise(values)}')
        print(f'  {"":<22} '
              f'{sign_split(values, "named BEFORE gold", "named AFTER gold")}')
        print('\n  If this is lopsided, a prompt can argue with it. The "name the'
              '\n  LAST mention" prompt already lost (0.737 vs 0.759), so check the'
              '\n  split before proposing it again in another form.')

    print('\n--- By question type ---')
    types: Dict[str, List[Dict]] = {}
    for f in scored:
        types.setdefault(f['row'].get('question_type', '?'), []).append(f)
    for name, group in sorted(types.items()):
        mean = sum(f['iou'] for f in group) / len(group)
        bad = sum(1 for f in group if f['cause'] == 'wrong-mention')
        print(f'  {name:<16} n={len(group):<5} mean tIoU {mean:.3f}   '
              f'wrong-mention {bad}')

    if worst:
        print(f'\n--- {worst} costliest ---')
        for f in sorted(scored, key=lambda f: f['iou'])[:worst]:
            print(f'\n  {f["row"]["question_id"]}  tIoU {f["iou"]:.2f} [{f["cause"]}]')
            print(f'    Q: {f["row"]["question"]}')
            print(f'    gold {f["gold"][0]:.2f}-{f["gold"][1]:.2f}   '
                  f'ours {f["span"][0]:.2f}-{f["span"][1]:.2f}')
            print(f'    quote: {str(f["quote"])[:110]!r}')
            print(f'    quote words found in gold: {f["quote_in_gold"]:.0%}')


def drift(directory: str) -> None:
    """Does the model behave the same on conversations we have no labels for?"""
    paths = sorted(Path(directory).glob('*.json'))
    if not paths:
        print(f'\nno request logs in {directory}')
        return
    lengths, unreadable, truncated, guessed = [], 0, 0, 0
    for path in paths:
        record = json.loads(path.read_text())
        segments = record.get('transcript') or []
        speech = segments[-1]['end'] if segments else 0.0
        audio = record.get('audio_seconds') or 0.0
        if speech and audio and speech < audio - 2.0:
            truncated += 1
        exchanges = record.get('exchanges') or []
        guessed += max(0, len(record.get('questions') or []) - len(exchanges))
        for exchange in exchanges:
            answer, _, quote = parse_reply(exchange.get('reply') or '')
            if answer is None:
                unreadable += 1
            if quote:
                lengths.append(len(_normalise(quote).split()))
    print(f'\n--- Unlabelled attempt: {len(paths)} conversations ---')
    print(f'  truncated transcripts   {truncated}')
    print(f'  questions never asked   {guessed}')
    print(f'  unreadable replies      {unreadable}')
    print(f'  quote length (words)    {summarise([float(v) for v in lengths], "")}')
    print('\n  Compare quote length against the labelled run below it. A shift'
          '\n  means the model is behaving differently on the unseen audio, which'
          '\n  is the only thing these logs can tell us without annotations.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replies', default=str(REPLIES))
    parser.add_argument('--worst', type=int, default=0)
    parser.add_argument('--requests', default='', help='a REQUEST_LOG_DIR to compare')
    args = parser.parse_args()

    found = collect(Path(args.replies))
    report(found, args.worst)
    quotes = [float(len(_normalise(f['quote']).split()))
              for f in found if f.get('quote')]
    print(f'\n  quote length here (words)  {summarise(quotes, "")}')
    if args.requests:
        drift(args.requests)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
