"""Re-time the cached transcripts with WhisperX forced alignment. Dev only.

Our word timings come from faster-whisper, which reads them off the decoder's
cross-attention -- a by-product of transcription, quantised to its own frame
rate. Forced alignment instead runs a phoneme CTC model (wav2vec2) over the
audio with the text already known, which is a much easier problem and gives
sharper bounds.

Whether that is worth anything *here* is a ceiling question, not a faith
question: `tools/score.py --ceiling-words` says the best word run our current
timings can express is worth 0.929 mean tIoU, so the honest test is whether the
same measurement rises once the timings are aligned. This tool produces the
re-timed transcripts; `tools/score.py --ceiling-words` scores them.

**It runs in its own virtualenv.** WhisperX pulls torch, pyannote and a second
copy of faster-whisper; none of that belongs anywhere near the attempt, so it
lives in `.venv-align` and writes plain JSON that the main venv reads back:

    ./.venv-align/bin/python tools/align.py
    TRANSCRIPTS_DIR=transcripts_aligned ./.venv/bin/python tools/score.py --ceiling-words

Segment text, index and bounds are copied through untouched. Only the word
timings change, which is what keeps the comparison a comparison.

Two ways the re-cut goes wrong, both of which produce a file that looks right:

**Do not zip the result against the input.** WhisperX re-splits the transcript
into one segment per sentence, so it hands back more segments than it was given
-- 52 for 26 on the first conversation -- and pairing them positionally gives
each segment an earlier segment's timings. That scores 0.485 where the truth is
above 0.9, and nothing in the output looks out of place.

**Do not cut the flat word list by count either.** The two tokenisers disagree
on hyphens: faster-whisper writes "anti" + "-inflammatory" where alignment
writes "anti-inflammatory". One such word shifts every timing after it. So the
streams are reconciled on their letters, and a word that alignment merged has
its span split back across the tokens we started with.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import transcripts  # noqa: E402

AUDIO = ROOT / 'data' / 'audio'
OUT = ROOT / 'transcripts_aligned'

# None lets WhisperX pick its default for the language -- wav2vec2-base-960h for
# English. If the ceiling moves at all it will move here, and cheaply; a larger
# alignment model is only worth reaching for once this one has shown something.
ALIGN_MODEL = None
LANGUAGE = 'en'
DEVICE = 'cpu'

Timing = Optional[Tuple[float, float]]


def _letters(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(text).casefold())


def recut(originals: List[Dict[str, Any]], aligned: List[Dict[str, Any]]) -> List[Timing]:
    """One timing per transcribed word token, taken from the aligned stream.

    The two streams carry the same letters in the same order but not always the
    same tokens, so they are walked together and the run of tokens on each side
    that spells the same thing is matched as a unit. Where alignment merged our
    tokens, the merged span is divided between them in proportion to their
    letters; where it split one of ours, the outer bounds are used.

    Returns ``None`` for any token it could not place, which the caller fills
    from the neighbours rather than dropping -- a short word stream would
    quietly change what the span logic can point at.
    """
    times: List[Timing] = [None] * len(originals)
    i = j = 0
    while i < len(originals) and j < len(aligned):
        if not _letters(originals[i].get('word', '')):
            i += 1
            continue
        if not _letters(aligned[j].get('word', '')):
            j += 1
            continue

        first_original, first_aligned = i, j
        ours = _letters(originals[i]['word'])
        theirs = _letters(aligned[j]['word'])
        while ours != theirs:
            if len(ours) < len(theirs):
                i += 1
                if i >= len(originals):
                    raise SystemExit('word streams cannot be reconciled')
                ours += _letters(originals[i]['word'])
            else:
                j += 1
                if j >= len(aligned):
                    raise SystemExit('word streams cannot be reconciled')
                theirs += _letters(aligned[j]['word'])

        start = aligned[first_aligned].get('start')
        end = aligned[j].get('end')
        if start is not None and end is not None:
            start, end = float(start), float(end)
            spelled = [len(_letters(originals[k]['word']))
                       for k in range(first_original, i + 1)]
            total = sum(spelled) or 1
            cursor = start
            for offset, length in enumerate(spelled):
                nxt = cursor + (end - start) * (length / total)
                times[first_original + offset] = (cursor, nxt)
                cursor = nxt
            times[i] = (times[i][0], end)
        i += 1
        j += 1
    return times


def fill_gaps(words: List[Dict[str, Any]], bounds: Tuple[float, float]) -> List[Dict[str, Any]]:
    """Give every word a time, including the ones alignment could not place.

    A word the CTC model cannot find comes back without bounds. Dropping it
    would silently shorten the word stream the span logic walks, so instead it
    inherits the gap between its placed neighbours.
    """
    start, end = bounds
    if not any(w.get('start') is not None for w in words):
        return [dict(w, start=start, end=end) for w in words]

    filled = []
    for n, word in enumerate(words):
        if word.get('start') is not None and word.get('end') is not None:
            filled.append(word)
            continue
        before = next((w for w in reversed(words[:n]) if w.get('end') is not None), None)
        after = next((w for w in words[n + 1:] if w.get('start') is not None), None)
        left = float(before['end']) if before else start
        right = float(after['start']) if after else end
        filled.append(dict(word, start=left, end=max(left, right)))
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()

    import whisperx
    from faster_whisper.audio import decode_audio

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE, device=DEVICE, model_name=ALIGN_MODEL)
    print(f'alignment model loaded in {time.perf_counter() - started:.1f} s', flush=True)

    paths = sorted(AUDIO.glob('conversation_*.mp3'))[:args.limit]
    total_audio, total_align, unplaced, total_words = 0.0, 0.0, 0, 0

    for path in paths:
        cached = transcripts.load(path.name)
        if cached is None:
            print(f'  {path.name}: no cached transcript, skipped', flush=True)
            continue

        audio = decode_audio(str(path), sampling_rate=16000)
        began = time.perf_counter()
        result = whisperx.align(
            [{'start': float(s['start']), 'end': float(s['end']), 'text': s['text']}
             for s in cached],
            model, metadata, audio, DEVICE, return_char_alignments=False,
        )
        took = time.perf_counter() - began

        originals = [w for segment in cached for w in (segment.get('words') or [])]
        times = recut(originals, list(result['word_segments']))

        segments, cursor = [], 0
        for segment in cached:
            words = []
            for word in (segment.get('words') or []):
                timing = times[cursor]
                cursor += 1
                total_words += 1
                if timing is None:
                    unplaced += 1
                    words.append({'word': word['word'], 'start': None, 'end': None})
                else:
                    words.append({'word': word['word'],
                                  'start': timing[0], 'end': timing[1]})
            words = fill_gaps(words, (float(segment['start']), float(segment['end'])))
            segments.append({
                'index': segment['index'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'words': [{'start': float(w['start']), 'end': float(w['end']),
                           'word': str(w['word'])} for w in words],
            })

        name = transcripts.path_for(path.name).name
        (out_dir / name).write_text(json.dumps({
            'audio_filename': path.name,
            'segments': segments,
            'aligned_with': 'whisperx forced alignment',
            'align_seconds': round(took, 2),
        }, indent=2))
        seconds = len(audio) / 16000
        total_audio += seconds
        total_align += took
        print(f'  {name}  {seconds:6.1f} s audio  aligned in {took:5.1f} s', flush=True)

    print(f'\n{len(paths)} conversations, {total_audio:.0f} s of audio')
    print(f'alignment took {total_align:.0f} s total, {total_align / max(len(paths), 1):.1f} s '
          f'each, {total_audio / max(total_align, 1e-9):.1f}x realtime')
    print(f'words alignment could not place: {unplaced} of {total_words} '
          f'({unplaced / max(total_words, 1):.2%}) -- filled from their neighbours')
    print(f'\nwritten to {out_dir}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
