"""How much of the loss is returning *too much*, and what could fix it. Dev only.

`tools/disagreements.py` says the error is lopsided towards over-inclusion.  This
asks the question that follows: **is "extract less" a lever, and how big is it at
most?**  It answers in three parts, all from the frozen replies, in under a
second:

1. **Attribution.**  Two counterfactuals: keep only the seconds inside gold
   (perfect trimming), and cover all of gold (perfect extension).  Their sizes
   say which half of the error is worth arguing with.

2. **Where the extra seconds are.**  A cross-tab of how many sentences we return
   against how many the annotation touches, which separates "one sentence too
   many" from "the right sentence, a little too wide".  The two need different
   fixes and only one of them is reachable from a prompt.

3. **Ceilings and gates.**  What a *perfect* extent signal would be worth -- the
   bound on any prompt field, second pass or heuristic that tries to decide how
   far a passage runs -- and what the obvious gates actually score.

Everything is over the 39 supplied conversations, the only annotated ones.

    python tools/extent.py
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import answering  # noqa: E402
import transcripts  # noqa: E402
from answering import (  # noqa: E402
    Sentence, _find_segment, _normalise, _overlap, _words_of, parse_reply,
    resolve_span, sentences_of,
)
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'

# A sentence merely abutting a span is not part of it -- the same threshold the
# span logic uses, so "how many sentences" means the same thing here as there.
TOUCH = 0.05


def quote_span(segment, quote: Optional[str]) -> Optional[Span]:
    """Where the quote lands, before it is rounded out to sentences.

    The same longest-run match :func:`answering.resolve_span` does, lifted out
    so the raw anchor can be measured on its own.
    """
    words = _words_of(segment)
    wanted = _normalise(quote).split() if quote else []
    if not words or not wanted:
        return None
    spoken = [_normalise(word['word']) for word in words]
    for length in range(len(wanted), 0, -1):
        for offset in range(len(wanted) - length + 1):
            needle = wanted[offset:offset + length]
            for start in range(len(spoken) - length + 1):
                if spoken[start:start + length] == needle:
                    return (float(words[start]['start']),
                            float(words[start + length - 1]['end']))
    return None


def touching(sentences: Sequence[Sentence], span: Span) -> List[Sentence]:
    return [s for s in sentences if _overlap(span, s) > TOUCH]


class Case:
    """One annotated question, with everything the report needs already found."""

    def __init__(self, row, sentences, segment, quote):
        self.row = row
        self.sentences = sentences
        self.gold: Span = gold_evidence(row)
        self.anchor: Span = (quote_span(segment, quote)
                             or (float(segment['start']), float(segment['end'])))
        self.quote_matched = quote_span(segment, quote) is not None
        self.predicted: Span = resolve_span(segment, quote, sentences, row['question'])
        self.iou = temporal_iou(self.gold, self.predicted)

    @property
    def gold_sentences(self) -> int:
        return len(touching(self.sentences, self.gold))

    @property
    def our_sentences(self) -> int:
        return len(touching(self.sentences, self.predicted))

    @property
    def quote_sentences(self) -> int:
        return len(touching(self.sentences, self.anchor)) if self.quote_matched else 0


def collect() -> List[Case]:
    replies = json.loads(REPLIES.read_text())
    found: List[Case] = []
    for audio_filename, rows in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = sentences_of(segments)
        for row in rows:
            text = replies.get(row['question_id'])
            if text is None or row['label'] != '1' or gold_evidence(row) is None:
                continue
            answer, index, quote = parse_reply(text)
            if answer is False:
                continue          # answered-no: a different cause, ticket 6
            segment = (_find_segment(segments, index)
                       or answering._best_keyword_segment(segments, row['question']))
            found.append(Case(row, sentences, segment, quote))
    return found


def attribution(cases: List[Case]) -> None:
    """Perfect trimming against perfect extension, in mean tIoU."""
    trim, extend, over_s, under_s = [], [], [], []
    for case in cases:
        (g0, g1), (p0, p1) = case.gold, case.predicted
        gold_len, our_len = g1 - g0, p1 - p0
        inter = max(0.0, min(g1, p1) - max(g0, p0))
        trim.append((inter / gold_len if gold_len else 0.0) - case.iou)
        extend.append((inter / our_len if our_len else 0.0) - case.iou)
        over_s.append(our_len - inter)
        under_s.append(gold_len - inter)

    print(f'\n--- What a perfect fix in each direction would be worth ---')
    print(f'  drop every second outside gold (perfect trimming)   '
          f'+{st.mean(trim):.3f} mean tIoU')
    print(f'  cover every second of gold (perfect extension)      '
          f'+{st.mean(extend):.3f} mean tIoU')
    print(f'\n  seconds we return that gold does not: median {st.median(over_s):.2f}  '
          f'total {sum(over_s):.0f}')
    print(f'  seconds of gold we never return:      median {st.median(under_s):.2f}  '
          f'total {sum(under_s):.0f}')
    clean = [o for o, u in zip(over_s, under_s) if u < TOUCH]
    print(f'  where gold is fully covered (n={len(clean)}), the extra we add: '
          f'median {st.median(clean):.2f} s')


def shape(cases: List[Case]) -> None:
    """Is the extra a whole sentence, or width inside the right one?"""
    pairs = Counter()
    ious = defaultdict(list)
    for case in cases:
        key = (case.our_sentences, case.gold_sentences)
        pairs[key] += 1
        ious[key].append(case.iou)

    print('\n--- Sentences we return against sentences gold touches ---')
    print(f'{"ours":>6}{"gold":>6}{"n":>6}{"mean tIoU":>11}')
    for key, count in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f'{key[0]:>6}{key[1]:>6}{count:>6}{st.mean(ious[key]):>11.3f}')
    more = sum(c for (o, g), c in pairs.items() if o > g)
    same = sum(c for (o, g), c in pairs.items() if o == g)
    fewer = sum(c for (o, g), c in pairs.items() if o < g)
    print(f'\n  we return more sentences than gold {more}   same {same}   fewer {fewer}')


def _score_with(cases: List[Case], max_run_of) -> float:
    """Mean tIoU when ``MAX_RUN_SENTENCES`` is chosen per question."""
    original = answering.MAX_RUN_SENTENCES
    try:
        ious = []
        for case in cases:
            answering.MAX_RUN_SENTENCES = max_run_of(case)
            span = answering.to_passage(case.anchor, case.sentences,
                                        case.row['question'])
            ious.append(temporal_iou(case.gold, span))
        return st.mean(ious)
    finally:
        answering.MAX_RUN_SENTENCES = original


def gates(cases: List[Case]) -> None:
    """What any signal that decides "one sentence or two" could be worth."""
    base = _score_with(cases, lambda c: 2)
    print(f'\n--- Deciding how far a passage runs (today: {base:.4f}) ---')
    for name, choose in (
        ('always one sentence', lambda c: 1),
        ('always three', lambda c: 3),
        ('as many as the quote spans, capped at two',
         lambda c: max(1, min(2, c.quote_sentences or 1))),
        ('**oracle**: as many as gold touches',
         lambda c: max(1, min(2, c.gold_sentences))),
    ):
        value = _score_with(cases, choose)
        print(f'  {name:<42} {value:.4f}   {value - base:+.4f}')

    one = [c for c in cases if c.quote_sentences == 1]
    many = [c for c in cases if c.quote_sentences >= 2]
    print(f'\n  gold is one sentence when the quote is one   '
          f'{sum(1 for c in one if c.gold_sentences == 1) / len(one):.0%}  (n={len(one)})')
    print(f'  gold is one sentence when the quote is two+  '
          f'{sum(1 for c in many if c.gold_sentences == 1) / len(many):.0%}  (n={len(many)})')
    print('  A gate needs these two to differ. They barely do, which is why'
          '\n  gating on the quote loses rather than wins.')

    # And the bound on choosing better among what the ranking already built.
    best, shorter, longer = [], 0, 0
    for case in cases:
        found = answering.passage_candidates(case.anchor, case.sentences,
                                             case.row['question'])
        if not found:
            best.append(case.iou)
            continue
        scored = [temporal_iou(case.gold, answering._finish(c.span, case.sentences))
                  for c in found]
        best.append(max(scored))
        pick = max(range(len(scored)), key=lambda i: scored[i])
        if scored[pick] - scored[0] < 0.02:
            continue
        top = found[0].span[1] - found[0].span[0]
        alternative = found[pick].span[1] - found[pick].span[0]
        shorter += alternative < top - TOUCH
        longer += alternative > top + TOUCH
    print(f'\n  **oracle**: pick the best of the shortlist          '
          f'{st.mean(best):.4f}   {st.mean(best) - base:+.4f}')
    print(f'  ...and where a better one existed it was shorter {shorter}, '
          f'longer {longer}.')
    print('  Not a width problem: a third of the reachable gain needs a *wider*'
          '\n  span, so a rule that only ever narrows cannot collect it.')


def main() -> int:
    cases = collect()
    print(f'{len(cases)} annotated questions answered yes, mean tIoU '
          f'{st.mean([c.iou for c in cases]):.4f}')
    attribution(cases)
    shape(cases)
    gates(cases)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
