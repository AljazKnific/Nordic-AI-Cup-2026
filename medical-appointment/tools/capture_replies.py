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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answering  # noqa: E402
import prompts  # noqa: E402
import transcripts  # noqa: E402
from answering import SELECT_TASK, answer_conversation  # noqa: E402
from utils import group_questions_by_conversation  # noqa: E402

DIAGNOSTICS = Path(__file__).resolve().parent.parent / 'diagnostics'
OUT = DIAGNOSTICS / 'replies.json'

# How a selection prompt is told from an answer prompt. Taken from SELECT_TASK
# itself rather than written out here: a literal copy goes stale the moment the
# prompt is re-worded, and the failure is silent -- selection replies would be
# filed as answers, advancing the cursor and misaligning every later reply.
_SELECT_MARKER = next(
    line for line in SELECT_TASK.splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--variant', default=None,
                        help=f'a prompt from tools/prompts.py: '
                             f'{", ".join(prompts.ALL)}')
    args = parser.parse_args()

    # The task text is swapped in memory rather than edited into answering.py,
    # so a capture cannot leave the shipped prompt changed behind it -- and so
    # two variants can be captured back to back without a stale .pyc between
    # them. The header still comes first, so the shared prefix holds (ADR-0001).
    if args.variant:
        variant = prompts.get(args.variant)
        answering.TASK = variant.task
        print(f'prompt: {variant.name}')
    out = Path(args.out) if args.out else (
        DIAGNOSTICS / f'replies_{args.variant}.json' if args.variant else OUT)
    args.out = str(out)

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

        # The second pass makes a *second* call for some questions. Advancing
        # the question order on those would file every later reply under the
        # wrong id and misalign the whole capture from that point -- silently,
        # because the file would still look complete. Only answer prompts move
        # the cursor; a selection reply is filed beside the question it belongs
        # to, under "<question_id>#select".
        current = {'id': None}

        def recording(prompt, timeout=None, _order=order, _rows=rows):
            selecting = _SELECT_MARKER in prompt
            text = ollama_client.answer(prompt)
            if selecting:
                if current['id'] is not None:
                    captured[f'{current["id"]}#select'] = text
                return text
            current['id'] = next(_order)
            captured[current['id']] = text
            return text

        answer_conversation(segments, [row['question'] for row in rows], recording)
        print(f'{audio_filename}  {time.perf_counter() - started:6.1f} s', flush=True)
        Path(args.out).write_text(json.dumps(captured, indent=1))

    Path(args.out).write_text(json.dumps(captured, indent=1))
    answers = sum(1 for k in captured if not k.endswith('#select'))
    picks = len(captured) - answers
    print(f'{answers} replies ({picks} second-pass choices) -> {args.out} '
          f'in {time.perf_counter() - started:.1f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
