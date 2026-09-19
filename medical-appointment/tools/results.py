"""Our answer and span beside the correct one, for every supplied question.

Not analysis -- just the numbers, so a person can look at them. Writes a CSV of
all 390 questions and prints the worst offenders. Runs off the frozen replies,
so it costs a second and needs no model.

    python tools/results.py                      # summary + worst 20
    python tools/results.py --worst 50
    python tools/results.py --csv results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import answering  # noqa: E402
import transcripts  # noqa: E402
from answering import _find_segment, parse_reply, resolve_span, sentences_of  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402

REPLIES = Path(__file__).resolve().parent.parent / 'diagnostics' / 'replies.json'
DEFAULT_CSV = Path(__file__).resolve().parent.parent / 'diagnostics' / 'results.csv'


def rows(replies_path: Path):
    replies = json.loads(replies_path.read_text())
    for audio_filename, questions in group_questions_by_conversation():
        segments = transcripts.load(audio_filename)
        if segments is None:
            continue
        sentences = sentences_of(segments)
        for row in questions:
            text = replies.get(row['question_id'])
            if text is None:
                continue
            answer, index, quote = parse_reply(text)
            span = None
            if answer is not False:
                segment = _find_segment(segments, index)
                if segment is None:
                    segment = answering._best_keyword_segment(segments, row['question'])
                span = resolve_span(segment, quote, sentences, row['question'])

            gold = gold_evidence(row)
            truth = row['answer'].strip().lower() == 'yes'
            ours = answer is not False
            yield {
                'question_id': row['question_id'],
                'conversation': row['transcript_id'],
                'type': row.get('question_type', ''),
                'question': row['question'],
                'correct_answer': 'yes' if truth else 'no',
                'our_answer': 'yes' if ours else 'no',
                'answer_ok': ours == truth,
                'correct_start': f'{gold[0]:.2f}' if gold else '',
                'correct_end': f'{gold[1]:.2f}' if gold else '',
                'our_start': f'{span[0]:.2f}' if span else '',
                'our_end': f'{span[1]:.2f}' if span else '',
                'tIoU': (f'{temporal_iou(gold, span):.3f}'
                         if gold and row['label'] == '1' else ''),
                'quote': (quote or '').replace('\n', ' '),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replies', default=str(REPLIES))
    parser.add_argument('--csv', default=str(DEFAULT_CSV))
    parser.add_argument('--worst', type=int, default=20)
    args = parser.parse_args()

    found = list(rows(Path(args.replies)))
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(found[0]))
        writer.writeheader()
        writer.writerows(found)

    scored = [r for r in found if r['tIoU']]
    wrong = [r for r in found if not r['answer_ok']]
    mean = sum(float(r['tIoU']) for r in scored) / len(scored)
    accuracy = sum(1 for r in found if r['answer_ok']) / len(found)
    print(f'{len(found)} questions   accuracy {accuracy:.3f}   '
          f'mean tIoU {mean:.3f}   SCORE {0.4 * accuracy + 0.6 * mean:.3f}')
    print(f'wrote {args.csv}\n')

    if wrong:
        print(f'--- {len(wrong)} WRONG ANSWERS ---')
        for r in wrong:
            print(f'  {r["question_id"]:<22} said {r["our_answer"]:<3} '
                  f'correct {r["correct_answer"]:<3}  {r["question"][:70]}')

    print(f'\n--- {args.worst} WORST SPANS (answered yes, annotated) ---')
    print(f'  {"question_id":<22}{"tIoU":>6}  {"correct":>14}  {"ours":>14}')
    for r in sorted(scored, key=lambda r: float(r['tIoU']))[:args.worst]:
        correct = f'{r["correct_start"]}-{r["correct_end"]}'
        ours = f'{r["our_start"]}-{r["our_end"]}' if r['our_start'] else 'no span'
        print(f'  {r["question_id"]:<22}{float(r["tIoU"]):>6.2f}  {correct:>14}  {ours:>14}')
        print(f'      Q: {r["question"][:78]}')
        if r['quote']:
            print(f'      quoted: {r["quote"][:78]!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
