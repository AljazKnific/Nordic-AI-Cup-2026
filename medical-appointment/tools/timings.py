"""Two sources of word timings, head to head against the annotation. Dev only.

`tools/score.py --ceiling-words` asks what the best run of words could score.
This asks the question behind it: **which timings express an annotated passage
better, and is the difference a real one or a constant offset?**

An offset is not nothing, but it is not new information either -- if one source
is simply shifted, a constant added to both edges recovers it, and the two
sources are then equivalent. So every ceiling here is reported raw *and* after
fitting the best constant, and the fit is in-sample on purpose: an upper bound
fitted on the same 195 passages is still an upper bound.

    ./.venv-align/bin/python tools/align.py     # writes transcripts_aligned/
    ./.venv/bin/python tools/timings.py

The last section is the one that decides it: how far an annotated edge sits from
the nearest word boundary each source offers. That is the quantisation the
ceiling is made of, and it does not care which run a chooser would pick.
"""

import argparse
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import transcripts  # noqa: E402
from utils import (  # noqa: E402
    Span, gold_evidence, group_questions_by_conversation, temporal_iou,
)

# Gold passages run a median of 2.9 s, so a run longer than ~30 words cannot be
# one; the cap keeps the quadratic scan cheap. Same bound tools/score.py uses.
MAX_RUN_WORDS = 30


def load(directory: str) -> List[Tuple[str, List[dict], List[dict]]]:
    """Every conversation's segments from one cache directory, with its rows."""
    original = transcripts.DIRECTORY
    transcripts.DIRECTORY = ROOT / directory
    try:
        found = []
        for audio_filename, rows in group_questions_by_conversation():
            segments = transcripts.load(audio_filename)
            if segments is not None:
                found.append((audio_filename, segments, rows))
        return found
    finally:
        transcripts.DIRECTORY = original


def words_of(segments: Sequence[dict]) -> List[dict]:
    return [w for segment in segments for w in (segment.get('words') or [])]


def best_word_span(words: Sequence[dict], gold: Span,
                   before: float = 0.0, after: float = 0.0) -> float:
    """The most any run of words could score, each edge moved outward by a constant."""
    best = 0.0
    for i in range(len(words)):
        start = float(words[i]['start']) - before
        for j in range(i, min(i + MAX_RUN_WORDS, len(words))):
            best = max(best, temporal_iou(gold, (start, float(words[j]['end']) + after)))
    return best


def ceiling(cases, before: float = 0.0, after: float = 0.0) -> float:
    return st.mean([best_word_span(words, gold, before, after) for words, gold in cases])


def cases_of(directory: str):
    found = []
    for _audio, segments, rows in load(directory):
        words = words_of(segments)
        for row in rows:
            gold = gold_evidence(row)
            if row['label'] == '1' and gold is not None and words:
                found.append((words, gold))
    return found


def fit_offsets(cases) -> Tuple[float, float, float]:
    """The constant outward move of each edge that scores best, and what it scores."""
    grid = [round(-0.4 + 0.05 * n, 2) for n in range(29)]     # -0.40 .. +1.00
    best = (0.0, 0.0, ceiling(cases))
    for before in grid:
        for after in grid:
            value = ceiling(cases, before, after)
            if value > best[2]:
                best = (before, after, value)
    return best


def edge_distances(cases) -> Tuple[List[float], List[float]]:
    """How far each annotated edge sits from the nearest word boundary on offer."""
    starts, ends = [], []
    for words, gold in cases:
        starts.append(min(abs(float(w['start']) - gold[0]) for w in words))
        ends.append(min(abs(float(w['end']) - gold[1]) for w in words))
    return starts, ends


def summarise(values: Sequence[float]) -> str:
    ordered = sorted(values)
    return (f'median {st.median(values):5.2f}s  mean {st.mean(values):5.2f}s  '
            f'p90 {ordered[int(0.9 * len(ordered))]:5.2f}s')


def shift(a: str, b: str) -> None:
    """Where the second source puts each word boundary, relative to the first."""
    left = {name: words_of(segs) for name, segs, _ in load(a)}
    right = {name: words_of(segs) for name, segs, _ in load(b)}
    starts, ends, durations_a, durations_b = [], [], [], []
    for name, words in left.items():
        other = right.get(name)
        if not other or len(other) != len(words):
            continue
        for one, two in zip(words, other):
            starts.append(float(two['start']) - float(one['start']))
            ends.append(float(two['end']) - float(one['end']))
            durations_a.append(float(one['end']) - float(one['start']))
            durations_b.append(float(two['end']) - float(two['start']))
    print(f'\n--- Where {b} puts a word, relative to {a} ({len(starts)} words) ---')
    print(f'  start moves   {summarise(starts)}')
    print(f'  end moves     {summarise(ends)}')
    print(f'  word duration {a}: median {st.median(durations_a):.2f}s   '
          f'{b}: median {st.median(durations_b):.2f}s')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dirs', nargs='+',
                        default=['transcripts', 'transcripts_aligned'])
    args = parser.parse_args()

    loaded = {name: cases_of(name) for name in args.dirs}
    print(f'{len(next(iter(loaded.values())))} annotated passages\n')
    print(f'{"timings":<24}{"raw":>10}{"calibrated":>13}{"before":>9}{"after":>9}')
    for name, cases in loaded.items():
        raw = ceiling(cases)
        before, after, best = fit_offsets(cases)
        print(f'{name:<24}{raw:>10.3f}{best:>13.3f}{before:>9.2f}{after:>9.2f}')
    print('\n  "calibrated" moves every word run outward by a fitted constant at each'
          '\n  edge, in-sample. It is the part of a difference that is only an offset.')

    if len(args.dirs) == 2:
        shift(*args.dirs)

    print('\n--- Distance from an annotated edge to the nearest word boundary ---')
    for name, cases in loaded.items():
        starts, ends = edge_distances(cases)
        print(f'  {name:<22} start  {summarise(starts)}')
        print(f'  {"":<22} end    {summarise(ends)}')
    print('\n  This is the quantisation the ceiling is made of. Finer timings can only'
          '\n  help if they put a boundary closer to where the annotation is drawn.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
