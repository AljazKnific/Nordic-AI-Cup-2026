"""Sequence-scoped tracking for frame-global detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    object_id: str
    bbox: BBox
    confidence: float


@dataclass
class Track:
    object_id: str
    bbox: BBox
    confidence: float
    last_frame: int
    velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def project(self, frame: int, confidence_decay: float) -> Detection:
        elapsed = max(0, frame - self.last_frame)
        projected = tuple(
            coordinate + rate * elapsed
            for coordinate, rate in zip(self.bbox, self.velocity)
        )
        confidence = self.confidence * (confidence_decay ** elapsed)
        return Detection(self.object_id, projected, confidence)


class SequenceTracker:
    """Track at most one active object per class for one sequence.

    The challenge reference data contains one instance per class. Keeping the
    state class-keyed prevents duplicate responses while still allowing a
    later direct observation to correct a projected box.
    """

    def __init__(
        self,
        confidence_decay: float = 0.96,
        minimum_confidence: float = 0.05,
        maximum_gap: int = 12,
    ) -> None:
        if not 0 < confidence_decay <= 1:
            raise ValueError('confidence_decay must be in (0, 1]')
        self.confidence_decay = confidence_decay
        self.minimum_confidence = minimum_confidence
        self.maximum_gap = maximum_gap
        self._sequence_id: Optional[str] = None
        self._tracks: Dict[str, Track] = {}

    def reset(self, sequence_id: Optional[str] = None) -> None:
        self._sequence_id = sequence_id
        self._tracks.clear()

    def update(
        self,
        sequence_id: str,
        frame: int,
        detections: Iterable[Detection],
    ) -> List[Detection]:
        if self._sequence_id != sequence_id:
            self.reset(sequence_id)

        for detection in self._deduplicate(detections):
            self._update_track(frame, detection)

        self._discard_stale(frame)
        return self.predict(frame)

    def predict(self, frame: int) -> List[Detection]:
        predictions = []
        for track in self._tracks.values():
            prediction = track.project(frame, self.confidence_decay)
            clipped = _clip_bbox(prediction.bbox)
            if clipped is None or prediction.confidence < self.minimum_confidence:
                continue
            predictions.append(
                Detection(prediction.object_id, clipped, prediction.confidence)
            )
        return sorted(predictions, key=lambda item: item.confidence, reverse=True)

    def _update_track(self, frame: int, detection: Detection) -> None:
        existing = self._tracks.get(detection.object_id)
        if existing is None:
            self._tracks[detection.object_id] = Track(
                detection.object_id,
                detection.bbox,
                detection.confidence,
                frame,
            )
            return

        elapsed = frame - existing.last_frame
        if elapsed > 0:
            velocity = tuple(
                (new - old) / elapsed
                for new, old in zip(detection.bbox, existing.bbox)
            )
        else:
            velocity = existing.velocity
        existing.bbox = detection.bbox
        existing.confidence = max(existing.confidence, detection.confidence)
        existing.last_frame = frame
        existing.velocity = velocity

    def _discard_stale(self, frame: int) -> None:
        stale = [
            object_id
            for object_id, track in self._tracks.items()
            if frame - track.last_frame > self.maximum_gap
            or track.project(frame, self.confidence_decay).confidence
            < self.minimum_confidence
        ]
        for object_id in stale:
            del self._tracks[object_id]

    @staticmethod
    def _deduplicate(detections: Iterable[Detection]) -> List[Detection]:
        best_by_class: Dict[str, Detection] = {}
        for detection in detections:
            bbox = _clip_bbox(detection.bbox)
            if bbox is None or not 0 <= detection.confidence <= 1:
                continue
            candidate = Detection(detection.object_id, bbox, detection.confidence)
            previous = best_by_class.get(candidate.object_id)
            if previous is None or candidate.confidence > previous.confidence:
                best_by_class[candidate.object_id] = candidate
        return list(best_by_class.values())


def _clip_bbox(bbox: Sequence[float]) -> Optional[BBox]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(value))) for value in bbox)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for two normalized boxes."""
    first_area = _area(first)
    second_area = _area(second)
    intersection = _area(
        (
            max(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            min(first[3], second[3]),
        )
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
