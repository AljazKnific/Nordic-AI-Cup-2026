# Medical Appointment

Answering ten yes/no questions about a recorded doctor–patient consultation from
the audio alone, and pointing at the passage each yes was read from. Scored as
`0.4 × Accuracy + 0.6 × mean tIoU`, so locating the evidence matters more than
answering.

## Language

**Conversation**:
One recorded consultation, arriving as a single MP3 with all ten of its questions
in one request. The unit of work: nothing is shared between conversations.
_Avoid_: File, clip, sample, audio

**Segment**:
A stretch of speech as the ASR chose to divide it, typically 7–8 seconds. An
artifact of transcription, not of the conversation.
_Avoid_: Chunk, block

**Mention**:
One place in a conversation where a fact is stated. A fact may be mentioned more
than once, and the annotated passage names only one of them.
_Avoid_: Occurrence, hit, match

**Passage**:
The stretch of speech that establishes an answer — around one sentence, median
2.9 seconds in the supplied data. Shorter than the segment containing it, and
what we are trying to point at.
_Avoid_: Excerpt, region, context

**Evidence span**:
The start and end second we return for a yes answer, our claim about where the
passage lies. Scored against the annotated one by temporal IoU.
_Avoid_: Timestamp, window, interval

**Quote**:
The words the answering model reports having read the answer off. Matched against
word timestamps to derive an evidence span.

**Positive**:
A question the conversation establishes as true. The only kind carrying an
annotated span, and the only kind counted in the tIoU average.

**Hard negative**:
A question that is a near-miss on something the conversation establishes — right
drug, wrong dose. Answerable only by reading precisely, never by topical overlap.

**Decoy term**:
An entity named by a hard negative that was never said in the conversation —
Pantoprazole where Esomeprazole was spoken. Arrives in the request looking exactly
like a real term, so it must never be fed back into transcription.
_Avoid_: Distractor, false term

**Question vocabulary**:
The entities named across a conversation's ten questions, correctly spelled and
available before transcription. Contains decoy terms as well as real ones, so it
is used to match what was heard, never to prime what is heard.

**Off topic**:
A question about a subject the conversation never raises. Answerable by absence.
