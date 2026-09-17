"""Transcribe the supplied conversations once and cache them. Dev only.

ASR is deterministic and the audio never changes, so re-transcribing on every
iteration costs ~13 s x 39 samples for nothing. Everything downstream -- prompt
wording, quote matching, span trimming -- only needs the segments.

    python tools/build_transcripts.py            # transcribe what is missing
    python tools/build_transcripts.py --force    # re-transcribe everything

Writes transcripts/<transcript_id>.json, which is gitignored. The request path
must never read these: the evaluation audio is unseen, so a cache hit there
would be a silent bug.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcripts  # noqa: E402
from asr import COMPUTE_TYPE, MODEL_NAME, transcribe  # noqa: E402
from utils import AUDIO_DIRECTORY  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='re-transcribe conversations that are already cached')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    transcripts.DIRECTORY.mkdir(exist_ok=True)

    audio_paths = sorted(AUDIO_DIRECTORY.glob('*.mp3'))
    if not audio_paths:
        logger.error('No audio found in %s', AUDIO_DIRECTORY)
        return 1

    transcribed = 0
    skipped = 0
    started = time.perf_counter()

    for audio_path in audio_paths:
        destination = transcripts.path_for(audio_path.name)
        if destination.exists() and not args.force:
            skipped += 1
            continue

        logger.info('%s', audio_path.name)
        segments = transcribe(audio_path.read_bytes())
        transcripts.save(audio_path.name, segments,
                         model=MODEL_NAME, compute_type=COMPUTE_TYPE)
        transcribed += 1

    logger.info(
        '\n%d transcribed, %d already cached, %.1f s total -> %s',
        transcribed, skipped, time.perf_counter() - started, transcripts.DIRECTORY,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
