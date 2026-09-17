"""Where cached transcripts live, and how they are named. Dev only.

One place, so the cache layout changes in one place. Both tools read it; the
request path never does -- the evaluation audio is unseen, so a cache hit there
would be a silent bug rather than a fast one.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DIRECTORY = Path(__file__).resolve().parent / 'transcripts'


def path_for(audio_filename: str) -> Path:
    """conversation_sample_17.mp3 -> transcripts/sample_17.json"""
    return DIRECTORY / (Path(audio_filename).stem.replace('conversation_', '') + '.json')


def load(audio_filename: str) -> Optional[List[Dict[str, Any]]]:
    """The cached segments for one conversation, or None if it has not been run."""
    path = path_for(audio_filename)
    if not path.exists():
        return None
    return json.loads(path.read_text())['segments']


def save(audio_filename: str, segments: List[Dict[str, Any]], **metadata: Any) -> Path:
    DIRECTORY.mkdir(exist_ok=True)
    path = path_for(audio_filename)
    path.write_text(json.dumps(
        {'audio_filename': audio_filename, 'segments': segments, **metadata}, indent=2,
    ))
    return path
