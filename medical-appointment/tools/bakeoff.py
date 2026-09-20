"""Score every captured prompt variant beside the shipped one. Dev only.

`tools/capture_replies.py --variant X` records one real reply per question under
that prompt; this scores all of them through the same span logic and prints the
table the decision is actually made on.

    ./.venv/bin/python tools/capture_replies.py --variant ends    # ~12 min each
    ./.venv/bin/python tools/bakeoff.py

Three things are reported per variant, because a prompt can lose in three
different places and the number alone does not say which:

* **score**, with accuracy and mean tIoU split out -- a prompt that trades one
  for the other is a different animal from one that is simply worse.
* **the cause breakdown**, the same attribution `tools/replay.py --diagnose`
  gives: whether the loss moved into naming the wrong mention, choosing the
  wrong sentences, or answering no.
* **how the anchor behaved** -- what fraction of replies gave an anchor that
  matched inside the named segment at all, and how wide it was. A variant that
  scores the same but anchors on half as many words has still told us
  something.

Only the annotated 39 conversations can be scored, and every variant is replayed
against the same transcripts, so the only thing that differs between rows is the
prompt.
"""

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answering  # noqa: E402
import prompts  # noqa: E402
import transcripts  # noqa: E402
from answering import _find_segment, sentences_of  # noqa: E402
from replay import best_sentence_run, sentences_in  # noqa: E402
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)

DIAGNOSTICS = Path(__file__).resolve().parent.parent / 'diagnostics'

# Where each variant's capture lives. The shipped prompt's capture is the plain
# one, because it was recorded before variants existed and re-recording it costs
# twelve minutes to reproduce byte for byte -- checked, on 2026-09-20, with
# `--limit 2`: 20 of 20 replies identical. The model is deterministic here.
CAPTURES = {
    'baseline': DIAGNOSTICS / 'replies.json',
    'ends': DIAGNOSTICS / 'replies_ends.json',
    'key': DIAGNOSTICS / 'replies_key.json',
    'proof': DIAGNOSTICS / 'replies_proof.json',
}


def score(name: str, path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    variant = prompts.get(name)
    replies: Dict[str, str] = json.loads(path.read_text())

    asked = correct = 0
    ious: List[float] = []
    causes: Dict[str, List[float]] = {}
    anchored, anchor_widths, unreadable = 0, [], 0
    # Per conversation, so the difference between two prompts can be resampled
    # over the unit that is actually independent. Questions inside one
    # conversation share a transcript and are not.
    per_conversation: Dict[str, Tuple[List[bool], List[float]]] = {}

    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = sentences_of(segments)
        marks, spans = per_conversation.setdefault(audio_filename, ([], []))
        for row in rows:
            text = replies.get(row['question_id'])
            if text is None:
                continue
            asked += 1
            answer, index, fields = variant.parse(text)
            if answer is None:
                unreadable += 1

            if answer is False:
                span: Optional[Span] = None
                said_yes = False
            else:
                said_yes = True
                named = _find_segment(segments, index)
                segment = named or answering._best_keyword_segment(
                    segments, row['question'])
                anchor = variant.anchor(segment, fields)
                if anchor:
                    anchored += 1
                    anchor_widths.append(anchor[1] - anchor[0])
                span = variant.locate(segment, fields, sentences, row['question'])

            correct += (said_yes == (row['label'] == '1'))
            marks.append(said_yes == (row['label'] == '1'))

            gold = gold_evidence(row)
            if row['label'] != '1' or gold is None:
                continue
            iou = temporal_iou(gold, span if said_yes else None)
            ious.append(iou)
            spans.append(iou)

            if not said_yes:
                cause, headroom = 'answered-no', best_sentence_run(sentences, gold)
            else:
                named = _find_segment(segments, index) or \
                    answering._best_keyword_segment(segments, row['question'])
                reachable = sentences_in(sentences, named)
                allowed = best_sentence_run(reachable, gold) if reachable else 0.0
                if allowed < 0.45:
                    cause, headroom = 'wrong-mention', best_sentence_run(sentences, gold)
                elif iou >= allowed - 0.03:
                    cause, headroom = 'best-available', iou
                else:
                    cause, headroom = 'wrong-sentences', allowed
            causes.setdefault(cause, []).append(max(0.0, headroom - iou))

    accuracy = correct / asked if asked else 0.0
    mean_iou = st.mean(ious) if ious else 0.0
    yes = sum(len(v) for k, v in causes.items() if k != 'answered-no')
    return {
        'name': name,
        'asked': asked,
        'accuracy': accuracy,
        'tiou': mean_iou,
        'score': 0.4 * accuracy + 0.6 * mean_iou,
        'causes': {k: sum(v) / len(ious) for k, v in causes.items()},
        'counts': {k: len(v) for k, v in causes.items()},
        'anchored': anchored / yes if yes else 0.0,
        'anchor_width': st.median(anchor_widths) if anchor_widths else 0.0,
        'unreadable': unreadable,
        'per_conversation': per_conversation,
    }


def bootstrap(a: Dict, b: Dict, draws: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """An interval on the score difference, resampling whole conversations.

    Paired: each draw takes the same conversations from both prompts, which is
    the comparison actually being made -- the two were asked the same questions
    about the same transcripts, and only the prompt differed. Resampling
    questions instead would treat ten questions about one consultation as ten
    independent observations, and report an interval several times too narrow.
    """
    names = sorted(set(a['per_conversation']) & set(b['per_conversation']))
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        drawn = [names[rng.randrange(len(names))] for _ in names]
        values = []
        for row in (a, b):
            marks = [m for name in drawn for m in row['per_conversation'][name][0]]
            spans = [s for name in drawn for s in row['per_conversation'][name][1]]
            accuracy = sum(marks) / len(marks) if marks else 0.0
            tiou = sum(spans) / len(spans) if spans else 0.0
            values.append(0.4 * accuracy + 0.6 * tiou)
        differences.append(values[1] - values[0])
    differences.sort()
    return differences[int(0.025 * draws)], differences[int(0.975 * draws)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', default=list(CAPTURES))
    args = parser.parse_args()

    rows = [r for r in (score(name, CAPTURES[name]) for name in args.variants) if r]
    if not rows:
        raise SystemExit('no captures found; run tools/capture_replies.py --variant X')
    base = rows[0]

    print(f'{"prompt":<12}{"n":>5}{"accuracy":>10}{"mean tIoU":>11}'
          f'{"score":>9}{"vs baseline":>13}{"95% CI":>22}')
    for row in rows:
        delta, interval = '', ''
        if row is not base:
            delta = f'{row["score"] - base["score"]:+.4f}'
            low, high = bootstrap(base, row)
            interval = f'{low:+.4f} to {high:+.4f}'
        print(f'{row["name"]:<12}{row["asked"]:>5}{row["accuracy"]:>10.3f}'
              f'{row["tiou"]:>11.4f}{row["score"]:>9.4f}{delta:>13}{interval:>22}')
    print('\n  The interval resamples whole conversations, paired: an interval'
          '\n  straddling zero means the two prompts are not distinguishable here.')

    print(f'\n{"":<12}{"recoverable tIoU by cause":>44}')
    order = ['wrong-sentences', 'wrong-mention', 'answered-no']
    print(f'{"prompt":<12}' + ''.join(f'{c:>18}' for c in order) + f'{"best-available":>17}')
    for row in rows:
        line = f'{row["name"]:<12}'
        for cause in order:
            line += f'{row["causes"].get(cause, 0.0):>13.3f} ({row["counts"].get(cause, 0):>2})'
        line += f'{row["counts"].get("best-available", 0):>17}'
        print(line)

    print(f'\n{"prompt":<12}{"anchor matched":>16}{"anchor width":>15}{"unreadable":>12}')
    for row in rows:
        print(f'{row["name"]:<12}{row["anchored"]:>15.1%}{row["anchor_width"]:>14.2f}s'
              f'{row["unreadable"]:>12}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
