"""The scored surface: wiring only.

The evaluator reaches this through ``api.py``. Everything interesting lives
elsewhere and is injected here — :mod:`asr` for transcription, :mod:`ollama_client`
for answering, :mod:`pipeline` for the contract, :mod:`answering` for the prompts,
parsing and spans. That split is what lets the whole thing be tested without a
model or audio; see tests/.

Both models are loaded and exercised at import, because there is no warm-up
period in the attempt and the first inference is the slowest one.
"""

import logging
from typing import Optional

import ollama_client
from asr import transcribe
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from pipeline import predict as run_pipeline

logger = logging.getLogger(__name__)

# asr imports load and exercise the ASR model. Do the same for the answering
# model, so the first real request pays for neither.
ollama_client.warm_up()


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(
    request: ASRQuestionRequestDto, arrived: Optional[float] = None,
) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    ``arrived`` comes from :mod:`api`, and is when the request landed rather
    than when its body finished parsing. Uploading a conversation is a real
    part of the evaluator's 60 s and has to be a real part of ours.
    """
    return run_pipeline(
        request, transcribe=transcribe, answerer=ollama_client.answer,
        arrived=arrived,
    )
