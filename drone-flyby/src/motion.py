"""Reference-scene motion fitting for carried-forward detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MotionObservation:
    """One source-frame observation of an object."""

    frame: int
    object_id: str
    bbox: BBox


@dataclass(frozen=True)
class _TrackFit:
    """Polynomial coefficients for one object's box components."""

    coefficients: Tuple[np.ndarray, ...]
    frame_origin: float

    def predict(self, frame: int) -> BBox:
        relative_frame = float(frame) - self.frame_origin
        values = [
            float(np.polyval(coefficient, relative_frame))
            for coefficient in self.coefficients
        ]
        x1, y1, x2, y2 = values
        return x1, y1, x2, y2


class MotionModel:
    """Fit smooth per-object box motion from annotated source frames.

    The model is deliberately independent from HTTP and image inference. It
    can therefore be fitted once at startup and used by the state layer for
    predictions when an object is outside the current camera view.
    """

    def __init__(self, fits: Dict[str, _TrackFit]):
        self._fits = fits

    @classmethod
    def fit(cls, observations: Iterable[MotionObservation]) -> "MotionModel":
        grouped: Dict[str, List[MotionObservation]] = {}
        for observation in observations:
            grouped.setdefault(observation.object_id, []).append(observation)

        fits: Dict[str, _TrackFit] = {}
        for object_id, track in grouped.items():
            track.sort(key=lambda item: item.frame)
            frames = np.asarray([item.frame for item in track], dtype=float)
            values = np.asarray([item.bbox for item in track], dtype=float)
            frame_origin = float(frames[0])
            relative_frames = frames - frame_origin
            degree = min(2, len(track) - 1)
            coefficients = tuple(
                np.polyfit(relative_frames, values[:, component], degree)
                for component in range(4)
            )
            fits[object_id] = _TrackFit(coefficients, frame_origin)
        return cls(fits)

    @classmethod
    def from_annotation_directory(cls, directory: Path) -> "MotionModel":
        """Fit a model from ``frame_*.json`` source-pixel annotations."""
        observations: List[MotionObservation] = []
        for path in sorted(directory.glob('frame_*.json')):
            payload = json.loads(path.read_text())
            frame = int(payload['frame'])
            for annotation in payload.get('annotations', []):
                bbox = tuple(float(value) for value in annotation['bbox'])
                if len(bbox) != 4:
                    raise ValueError(f'Expected four bbox values in {path}')
                observations.append(
                    MotionObservation(frame, annotation['object_id'], bbox)
                )
        return cls.fit(observations)

    @property
    def object_ids(self) -> Tuple[str, ...]:
        """Return the object classes represented by the fitted reference data."""
        return tuple(sorted(self._fits))

    def predict(self, object_id: str, frame: int) -> Optional[BBox]:
        """Predict a source-pixel box, or ``None`` for an unknown object."""
        fit = self._fits.get(object_id)
        return None if fit is None else fit.predict(frame)

    def predict_from_observation(
        self,
        observation: MotionObservation,
        target_frame: int,
    ) -> Optional[BBox]:
        """Forecast from a known observation using the fitted object track."""
        if observation.object_id not in self._fits:
            return None
        return self.predict(observation.object_id, target_frame)


def load_reference_motion(project_root: Path) -> MotionModel:
    """Load the bundled Helsinki reference motion model."""
    annotation_directory = project_root / 'src' / 'helsinki' / 'annotations'
    return MotionModel.from_annotation_directory(annotation_directory)
