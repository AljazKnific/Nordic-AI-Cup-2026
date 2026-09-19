"""Read the request logs an attempt wrote. Dev only.

`request_log.py` writes one JSON per conversation when REQUEST_LOG_DIR is set.
This reads them back: the transcript as the ASR heard it, and for each question
the raw reply, the quote, and the span we returned. It is the only view we get
of the evaluation conversations, which are otherwise unseen.

    python tools/inspect_requests.py diagnostics/requests            # summary
    python tools/inspect_requests.py diagnostics/requests --transcript
    python tools/inspect_requests.py diagnostics/requests --only sample_46
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answering import parse_reply  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--transcript', action='store_true',
                        help='print the transcript, segment by segment')
    parser.add_argument('--only', default='', help='substring of the filename')
    args = parser.parse_args()

    paths = sorted(Path(args.directory).glob('*.json'))
    if not paths:
        print(f'no request logs in {args.directory}')
        return 1

    for path in paths:
        record = json.loads(path.read_text())
        name = record.get('audio_filename', path.stem)
        if args.only and args.only not in name:
            continue

        timings = record.get('timings') or {}
        segments = record.get('transcript') or []
        speech = segments[-1]['end'] if segments else 0.0
        audio = record.get('audio_seconds') or 0.0
        print(f'\n=== {name}')
        if record.get('failed'):
            print(f'  FAILED after {record.get("seconds")} s')
        print(f'  audio {audio:.1f} s, transcribed to {speech:.1f} s '
              f'({len(segments)} segments)')
        if speech and audio and speech < audio - 2.0:
            print(f'  ** TRUNCATED: {audio - speech:.1f} s of audio never transcribed **')
        if timings:
            print('  timings ' + ', '.join(f'{k} {v}s' for k, v in timings.items()))

        if args.transcript:
            for segment in segments:
                print(f'    [{segment["index"]:>3}] {segment["start"]:>7.2f}-'
                      f'{segment["end"]:>7.2f}  {segment["text"]}')

        exchanges = record.get('exchanges') or []
        questions = record.get('questions') or []
        answers = record.get('answers') or []
        starts = record.get('evidence_start') or []
        ends = record.get('evidence_end') or []
        if not exchanges:
            continue
        print(f'  {len(exchanges)} of {len(questions)} questions reached the model')
        for i, exchange in enumerate(exchanges):
            reply = exchange.get('reply')
            if exchange.get('error'):
                print(f'    Q{i + 1} ERROR {exchange["error"]}  '
                      f'(given {exchange.get("timeout_given")} s)')
                continue
            answer, index, quote = parse_reply(reply or '')
            span = ''
            if i < len(starts) and starts[i] is not None:
                span = f'  span {starts[i]:.2f}-{ends[i]:.2f}'
            shown = 'yes' if (i < len(answers) and answers[i]) else 'no'
            print(f'    Q{i + 1} [{exchange["seconds"]:>5.2f}s] {shown:<3} '
                  f'seg={index}{span}')
            print(f'        {exchange["question"][:100]}')
            if quote:
                print(f'        quote: {quote[:100]!r}')
        guessed = len(questions) - len(exchanges)
        if guessed > 0:
            print(f'  ** {guessed} question(s) never asked: guessed past the deadline **')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
