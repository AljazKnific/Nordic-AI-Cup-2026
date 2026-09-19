"""A written record of what one request saw, for reading after an attempt.

The evaluation conversations are unseen: the only way to find out whether the
ASR heard them correctly, or which sentence the model quoted, is to write it
down while the request is in flight. `tools/build_transcripts.py` cannot help
here -- it works on the 39 supplied files, and the ones that matter are the ones
we have never had.

**This is write-only, deliberately.** `asr.py` notes that a transcript cache in
the request path would be a silent bug, since the evaluation audio is unseen and
a cache hit would serve the wrong conversation. Nothing here ever reads a record
back; it only appends. There is no path from this module into an answer.

Off unless ``REQUEST_LOG_DIR`` is set, and every write is wrapped: a diagnostic
that can fail a request is worth less than no diagnostic at all.

    REQUEST_LOG_DIR=diagnostics/requests caffeinate -dimsu ./.venv/bin/python api.py
    ./.venv/bin/python tools/inspect_requests.py diagnostics/requests
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DIRECTORY = os.environ.get('REQUEST_LOG_DIR', '')


def enabled() -> bool:
    return bool(DIRECTORY)


class Record:
    """What one conversation looked like on the way through."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.started = time.time()
        self.exchanges: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {'audio_filename': filename}

    def note(self, **fields: Any) -> None:
        self.data.update(fields)

    def recording(self, answerer):
        """Wrap the answerer so every prompt and raw reply is kept."""
        def recorded(prompt: str, timeout: Optional[float] = None) -> str:
            started = time.perf_counter()
            try:
                reply = answerer(prompt, timeout=timeout)
            except Exception as error:
                self.exchanges.append({
                    'question': _question_of(prompt),
                    'timeout_given': round(timeout, 2) if timeout else None,
                    'seconds': round(time.perf_counter() - started, 2),
                    'reply': None,
                    'error': f'{type(error).__name__}: {error}',
                })
                raise
            self.exchanges.append({
                'question': _question_of(prompt),
                'timeout_given': round(timeout, 2) if timeout else None,
                'seconds': round(time.perf_counter() - started, 2),
                'reply': reply,
                'error': None,
            })
            return reply
        return recorded

    def write(self) -> None:
        if not DIRECTORY:
            return
        try:
            os.makedirs(DIRECTORY, exist_ok=True)
            stem = os.path.splitext(os.path.basename(self.filename))[0] or 'request'
            path = os.path.join(DIRECTORY, f'{stem}-{int(self.started)}.json')
            self.data['exchanges'] = self.exchanges
            with open(path, 'w') as handle:
                json.dump(self.data, handle, indent=2, default=float)
            logger.info('LOG wrote %s', path)
        except Exception:
            # Never let bookkeeping cost a conversation ten marks.
            logger.exception('Could not write the request log; continuing')


def _question_of(prompt: str) -> str:
    """The question off the end of a prompt, without the transcript header."""
    _, _, tail = prompt.rpartition('Question: ')
    return tail.strip()


def transcript_of(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Segments as written down: text and bounds, with the words kept."""
    return [
        {
            'index': segment.get('index'),
            'start': segment.get('start'),
            'end': segment.get('end'),
            'text': segment.get('text'),
            'words': segment.get('words') or [],
        }
        for segment in segments
    ]
