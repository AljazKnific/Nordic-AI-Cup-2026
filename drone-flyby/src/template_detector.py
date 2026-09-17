"""OpenCV reference-template detector used when no neural stack is installed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from utils import clip_bbox_to_frame, view_bbox_to_global


@dataclass(frozen=True)
class Template:
    object_id: str
    image: np.ndarray


@dataclass(frozen=True)
class TemplateDetection:
    object_id: str
    bbox: Tuple[float, float, float, float]
    confidence: float


class ReferenceTemplateDetector:
    """Match reference object crops at the scale implied by the camera view.

    This is a deterministic fallback, not a replacement for a trained model.
    It keeps the complete local pipeline executable on CPU-only machines and
    provides a useful baseline for measuring camera and tracking behavior.
    """

    def __init__(
        self,
        templates: Sequence[Template],
        threshold: float = 0.62,
        scales: Sequence[float] = (1.0,),
    ) -> None:
        cv2.setNumThreads(1)
        self.templates = tuple(templates)
        self.threshold = threshold
        self.scales = tuple(scales)
        self._candidate_cache: Dict[Tuple[int, int, int, int], Tuple[Tuple[Template, np.ndarray], ...]] = {}

    @classmethod
    def from_reference_data(cls, project_root: Path) -> "ReferenceTemplateDetector":
        from .synthetic_data import load_reference_crops

        crops = load_reference_crops(
            project_root / 'src' / 'helsinki' / 'images',
            project_root / 'src' / 'helsinki' / 'annotations',
        )
        templates = [
            Template(object_id, cv2.cvtColor(crop.image, cv2.COLOR_BGR2GRAY))
            for object_id, crop in sorted(crops.items())
        ]
        return cls(templates)

    def warmup(self, image_shape: Tuple[int, int] = (540, 960)) -> None:
        image = np.zeros((*image_shape, 3), dtype=np.uint8)
        self.detect(image, (0, 0, 3840, 2160), 3840, 2160)

    def detect(
        self,
        image: np.ndarray,
        source_region_xyxy: Sequence[int],
        original_width: int,
        original_height: int,
        expected_boxes: Mapping[str, Sequence[float]] | None = None,
    ) -> List[TemplateDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        view_width = source_region_xyxy[2] - source_region_xyxy[0]
        view_height = source_region_xyxy[3] - source_region_xyxy[1]
        scale_x = image.shape[1] / float(view_width)
        scale_y = image.shape[0] / float(view_height)
        best_by_object: Dict[str, Tuple[float, Tuple[int, int, int, int]]] = {}

        cache_key = (image.shape[1], image.shape[0], view_width, view_height)
        candidates = self._candidate_cache.get(cache_key)
        if candidates is None:
            prepared = []
            for template in self.templates:
                for scale in self.scales:
                    resized_width = max(3, int(template.image.shape[1] * scale_x * scale))
                    resized_height = max(3, int(template.image.shape[0] * scale_y * scale))
                    if resized_width >= gray.shape[1] or resized_height >= gray.shape[0]:
                        continue
                    candidate = cv2.resize(
                        template.image,
                        (resized_width, resized_height),
                        interpolation=cv2.INTER_AREA if scale_x < 1 else cv2.INTER_LINEAR,
                    )
                    prepared.append((template, candidate))
            candidates = tuple(prepared)
            self._candidate_cache[cache_key] = candidates

        for template, candidate in candidates:
            search_image = gray
            offset_x = 0
            offset_y = 0
            if expected_boxes is not None:
                expected = expected_boxes.get(template.object_id)
                if expected is not None:
                    global_x1, global_y1, global_x2, global_y2 = expected
                    region_x1, region_y1, region_x2, region_y2 = source_region_xyxy
                    local_x1 = (global_x1 * original_width - region_x1) * scale_x
                    local_y1 = (global_y1 * original_height - region_y1) * scale_y
                    local_x2 = (global_x2 * original_width - region_x1) * scale_x
                    local_y2 = (global_y2 * original_height - region_y1) * scale_y
                    expected_width = max(candidate.shape[1], local_x2 - local_x1)
                    expected_height = max(candidate.shape[0], local_y2 - local_y1)
                    padding_x = max(20, int(expected_width * 1.5))
                    padding_y = max(20, int(expected_height * 1.5))
                    offset_x = max(0, int((local_x1 + local_x2) / 2 - padding_x))
                    offset_y = max(0, int((local_y1 + local_y2) / 2 - padding_y))
                    end_x = min(gray.shape[1], int((local_x1 + local_x2) / 2 + padding_x))
                    end_y = min(gray.shape[0], int((local_y1 + local_y2) / 2 + padding_y))
                    if end_x - offset_x <= candidate.shape[1] or end_y - offset_y <= candidate.shape[0]:
                        continue
                    search_image = gray[offset_y:end_y, offset_x:end_x]

            result = cv2.matchTemplate(search_image, candidate, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            best = (
                score,
                (location[0] + offset_x, location[1] + offset_y,
                 candidate.shape[1], candidate.shape[0]),
            )
            if best is None or best[0] < self.threshold:
                continue
            previous = best_by_object.get(template.object_id)
            if previous is None or best[0] > previous[0]:
                best_by_object[template.object_id] = best

        detections: List[TemplateDetection] = []
        for object_id, (score, (x, y, width, height)) in best_by_object.items():
            local_bbox = (
                x / image.shape[1],
                y / image.shape[0],
                (x + width) / image.shape[1],
                (y + height) / image.shape[0],
            )
            global_bbox = clip_bbox_to_frame(
                view_bbox_to_global(
                    local_bbox,
                    source_region_xyxy,
                    original_width,
                    original_height,
                )
            )
            if global_bbox is not None:
                detections.append(
                    TemplateDetection(object_id, global_bbox, float(score))
                )
        return detections
