"""Capture one raw model reply per question, so span work can iterate offline.

Dev only. A full scoring run costs ~12 minutes because it re-asks the model; the
span logic under test does not depend on the model at all. This runs the real
sequential, shared-prefix path once and writes every raw reply to
``diagnostics/replies.json``, keyed by question id. ``tools/replay.py`` then
scores any span strategy against those replies in under a second.

The deadline is lifted for the capture so every question yields a reply; the
replay can simulate a deadline if it ever needs to.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('ANSWER_DEADLINE', '100000')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcripts  # noqa: E402
from answering import answer_conversation  # noqa: E402
from utils import group_questions_by_conversation  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()

    import ollama_client
    ollama_client.warm_up()

    groups = group_questions_by_conversation()
    if args.limit:
        groups = groups[:args.limit]

    captured = {}
    started = time.perf_counter()
    for audio_filename, rows in groups:
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        pending = [row['question_id'] for row in rows]
        order = iter(pending)

        def recording(prompt, _order=order, _rows=rows):
            text = ollama_client.answer(prompt)
            captured[next(_order)] = text
            return text

        answer_conversation(segments, [row['question'] for row in rows], recording)
        print(f'{audio_filename}  {time.perf_counter() - started:6.1f} s', flush=True)
        Path(args.out).write_text(json.dumps(captured, indent=1))

    Path(args.out).write_text(json.dumps(captured, indent=1))
    print(f'{len(captured)} replies -> {args.out} in {time.perf_counter() - started:.1f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
