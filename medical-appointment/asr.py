"""Local transcription with the timings kept.

The span we return for a yes answer is derived from these timestamps, so timing
is part of the answer here, not a means to an end. Every segment carries its
word-level timings; :mod:`answering` narrows a segment down to the passage using
them.

The model is loaded *and exercised* at import, because there is no warm-up
period in the attempt and the first inference is the slowest one.

Nothing in here touches the transcript cache. The cache exists for the dev loop
only (see ``tools/build_transcripts.py``); a cache hit in the request path would
be a silent bug, since the evaluation audio is unseen.
"""

import logging
import os
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# distil-large-v3 int8 on CPU, measured at ~5x realtime on this machine. There
# is no CUDA here, so float16 and full large-v3 are not on the table. Changing
# either of these changes the budget: re-measure with tools/score.py before and
# after, since ASR is the larger half of the per-conversation cost.
MODEL_NAME = os.environ.get('ASR_MODEL', 'distil-large-v3')
DEVICE = os.environ.get('ASR_DEVICE', 'cpu')
COMPUTE_TYPE = os.environ.get('ASR_COMPUTE_TYPE', 'int8')

# ctranslate2 defaults to a conservative thread count and leaves most of the
# machine idle. Measured on the longest supplied conversation (232 s of audio):
# 40.4 s at the default against 27.9 s across 12 threads, for identical output.
# ASR is ~70% of the per-conversation budget, so this is the difference between
# a long conversation fitting in 60 s and timing out -- and a timeout costs all
# ten of its marks. Scale to the host rather than to this laptop.
CPU_THREADS = int(os.environ.get('ASR_CPU_THREADS', '0')) or (os.cpu_count() or 4)

# Keep weights next to the project so the attempt never needs the network.
DOWNLOAD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _load_model() -> WhisperModel:
    started = time.perf_counter()
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=DOWNLOAD_ROOT,
        cpu_threads=CPU_THREADS,
    )
    logger.info(
        'ASR model %s (%s/%s, %d threads) loaded in %.1f s',
        MODEL_NAME, DEVICE, COMPUTE_TYPE, CPU_THREADS, time.perf_counter() - started,
    )
    return model


_MODEL = _load_model()


def _warm_up() -> None:
    """Pay for the first inference now rather than during the first request."""
    started = time.perf_counter()
    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = _MODEL.transcribe(silence, language='en', word_timestamps=True)
    list(segments)
    logger.info('ASR warm-up in %.1f s', time.perf_counter() - started)


_warm_up()


def transcribe(audio_bytes: bytes) -> List[Dict[str, Any]]:
    """Transcribe one conversation into numbered segments with word timings.

    Returns a list of ``{index, start, end, text, words}`` dicts, where each word
    is ``{start, end, word}``. The ``index`` is what the answering model names
    when it reports which segment it read an answer off, and it is the key the
    quote match is scoped to.
    """
    started = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix='.mp3') as handle:
        handle.write(audio_bytes)
        handle.flush()
        raw_segments, info = _MODEL.transcribe(
            handle.name,
            language='en',
            word_timestamps=True,
        )

        segments: List[Dict[str, Any]] = []
        for index, segment in enumerate(raw_segments):
            segments.append({
                'index': index,
                'start': round(float(segment.start), 2),
                'end': round(float(segment.end), 2),
                'text': segment.text.strip(),
                'words': [
                    {
                        'start': round(float(word.start), 2),
                        'end': round(float(word.end), 2),
                        'word': word.word.strip(),
                    }
                    for word in (segment.words or [])
                ],
            })

    elapsed = time.perf_counter() - started
    speech = segments[-1]['end'] if segments else 0.0
    logger.info(
        'ASR: %d segments over %.1f s of speech in %.1f s (%.1fx realtime)',
        len(segments), speech, elapsed, speech / elapsed if elapsed else 0.0,
    )
    return segments
