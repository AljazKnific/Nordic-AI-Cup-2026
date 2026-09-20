"""Prompt variants to capture and score against the shipped one. Dev only.

Six prompt experiments have been run on this case and none of them shipped (see
`docs/tickets.md`), so the bar for a seventh is a *mechanism the earlier ones
did not test* -- not another wording of "quote less", which lost three ways.

Each variant here is one call per question, the same schema-shaped reply, and
the same byte-identical shared prefix (ADR-0001): only the task text after the
transcript changes, and with it what we anchor the passage on.

    ./.venv/bin/python tools/capture_replies.py --variant ends
    ./.venv/bin/python tools/replay.py --variant ends --replies diagnostics/replies_ends.json

`baseline` is the shipped prompt, re-declared here so a fresh capture of it can
be scored beside the others rather than compared against a capture from another
day and another commit.

The variants, and what each is actually testing:

**ends** -- two anchors instead of one. The model marks where the passage
*begins* and where it *ends*, and we span between them. Everything tried so far
has asked for one quote and then guessed its extent with a reach and a run
cap; this asks the model for the extent directly, in the only terms it can give
reliably -- words it can see. A perfect extent signal is worth +0.026 mean tIoU
(`tools/extent.py`), and this is the cheapest way to ask for one.

**key** -- the quote, plus the shortest phrase inside it that states the answer.
The single quote field does two jobs today: it localises (which wants length,
because the longest-run matcher is more reliable with more words) and it anchors
the ranking (which wants tightness). Asking for both separately is the one way
to stop those two pulling against each other, and it is why "quote one sentence"
lost -- it bought tightness by giving up localisation.

**proof** -- the same single quote, reframed as what a reviewer would be shown.
The cheap arm: no schema change, no extra tokens, and it tests whether the model
reads "the words that answer it" as the minimal fact when the annotation is a
whole spoken passage. If the two structural arms move nothing and this does,
the problem was never structural.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answering import (  # noqa: E402
    Sentence, locate_quote, segment_bounds, to_passage,
)
from utils import Span  # noqa: E402

# Every variant opens with the same adversarial warning. Accuracy is 0.990 and
# worth +0.004 in total, so no experiment here is allowed to put it at risk:
# what changes is the evidence half of the reply, never the answering half.
PREAMBLE = """
Answer the question below using only the consultation above.

The questions are adversarial: many are near-misses on something the
consultation does establish -- the right drug at the wrong dose, the right
course at the wrong length. Compare every number, unit, drug name and duration
against the transcript before answering yes. If the question differs from the
transcript in any detail, the answer is no. If the subject never comes up at
all, the answer is no.

Reply with JSON only:
"""

TAIL = '\nQuestion: '


def _payload(text: str) -> Tuple[Optional[bool], Any, Dict[str, Any]]:
    """Answer, segment and every other field, out of one reply.

    Same layering as :func:`answering.parse_reply` -- strict JSON, then a bare
    yes/no, then nothing -- so a variant is never scored against a more
    forgiving parser than the shipped prompt gets.
    """
    if not text:
        return None, None, {}

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            found = json.loads(match.group(0))
            answer = found.get('answer')
            if isinstance(answer, str):
                answer = answer.strip().casefold().startswith('y')
            if isinstance(answer, bool):
                return answer, found.get('segment'), found
        except (ValueError, AttributeError):
            pass

    leading = re.match(r'\W*(yes|no)\b', text.strip(), re.IGNORECASE)
    if leading:
        return leading.group(1).casefold() == 'yes', None, {}
    return None, None, {}


def _text(fields: Dict[str, Any], name: str) -> Optional[str]:
    value = fields.get(name)
    return value if isinstance(value, str) and value.strip() else None


class Variant(NamedTuple):
    name: str
    task: str
    # (segment, fields) -> where in that segment the passage is anchored, or
    # None to fall back to the segment's own bounds.
    anchor: Callable[[Dict[str, Any], Dict[str, Any]], Optional[Span]]

    def parse(self, text: str) -> Tuple[Optional[bool], Any, Dict[str, Any]]:
        return _payload(text)

    def locate(self, segment: Dict[str, Any], fields: Dict[str, Any],
               sentences: Sequence[Sentence] = (),
               question: Optional[str] = None) -> Span:
        """The finished passage, through the same span logic as the shipped path."""
        span = self.anchor(segment, fields) or segment_bounds(segment)
        return to_passage(span, sentences, question)


def _baseline_anchor(segment, fields):
    return locate_quote(segment, _text(fields, 'quote'))


def _ends_anchor(segment, fields):
    """Between the two marked ends -- or on whichever one of them matched.

    A single end is still a better anchor than the whole segment, and it is the
    common failure: the model marks the opening words of the passage correctly
    and paraphrases the closing ones.
    """
    first = locate_quote(segment, _text(fields, 'first_words'))
    last = locate_quote(segment, _text(fields, 'last_words'))
    if first and last:
        return (min(first[0], last[0]), max(first[1], last[1]))
    return first or last


def _key_anchor(segment, fields):
    """The key phrase, falling back to the quote it was drawn from.

    The quote is the safety net on purpose: if the model invents a key phrase
    that is not in the segment, this variant must not score worse than the
    prompt it is being compared against for a reason that has nothing to do
    with what it is testing.
    """
    return (locate_quote(segment, _text(fields, 'key'))
            or locate_quote(segment, _text(fields, 'quote')))


BASELINE = Variant(
    name='baseline',
    task=PREAMBLE + """{"answer": "yes" or "no", "segment": <the number of the segment you read the answer off, as a plain integer with no brackets>, "quote": "<the exact words from that segment that answer it>"}

For "no", use {"answer": "no", "segment": null, "quote": null}.
The quote must be copied exactly from the segment, not paraphrased.
""" + TAIL,
    anchor=_baseline_anchor,
)

ENDS = Variant(
    name='ends',
    task=PREAMBLE + """{"answer": "yes" or "no", "segment": <the number of the segment you read the answer off, as a plain integer with no brackets>, "first_words": "<the words the passage starts with>", "last_words": "<the words it ends with>"}

For "no", use {"answer": "no", "segment": null, "first_words": null, "last_words": null}.
Mark the whole stretch of the consultation that establishes the answer, not
only the detail asked about: first_words are the two or three words it begins
with and last_words the two or three words it ends with. Both must be copied
exactly from that segment.
""" + TAIL,
    anchor=_ends_anchor,
)

KEY = Variant(
    name='key',
    task=PREAMBLE + """{"answer": "yes" or "no", "segment": <the number of the segment you read the answer off, as a plain integer with no brackets>, "quote": "<the exact words from that segment that answer it>", "key": "<the shortest phrase inside that quote that states the answer>"}

For "no", use {"answer": "no", "segment": null, "quote": null, "key": null}.
Both must be copied exactly from the segment, not paraphrased, and the key must
appear inside the quote.
""" + TAIL,
    anchor=_key_anchor,
)

PROOF = Variant(
    name='proof',
    task=PREAMBLE + """{"answer": "yes" or "no", "segment": <the number of the segment you read the answer off, as a plain integer with no brackets>, "quote": "<the words from that segment that prove it>"}

For "no", use {"answer": "no", "segment": null, "quote": null}.
The quote is what someone checking your answer would be shown: the continuous
stretch of what was said that establishes it, as it would be read aloud, rather
than the single detail lifted out of it. Copy it exactly from the segment.
""" + TAIL,
    anchor=_baseline_anchor,
)

ALL = {v.name: v for v in (BASELINE, ENDS, KEY, PROOF)}


def get(name: str) -> Variant:
    if name not in ALL:
        raise SystemExit(f'unknown variant {name!r}; have {", ".join(ALL)}')
    return ALL[name]
