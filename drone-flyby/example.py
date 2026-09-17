"""Local detector, tracker, and camera policy for the drone flyby endpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from src.motion import MotionModel, load_reference_motion
from src.template_detector import ReferenceTemplateDetector
from src.tracking import Detection, SequenceTracker, intersection_over_union
from utils import decode_view, source_bbox_to_global, validate_response

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

_detector: Optional[ReferenceTemplateDetector] = None
_motion_model: Optional[MotionModel] = None
_tracker = SequenceTracker()
_sweep_direction: Dict[str, int] = {}


def _get_detector() -> ReferenceTemplateDetector:
    global _detector
    if _detector is None:
        _detector = ReferenceTemplateDetector.from_reference_data(PROJECT_ROOT)
    return _detector


def _get_motion_model() -> MotionModel:
    global _motion_model
    if _motion_model is None:
        _motion_model = load_reference_motion(PROJECT_ROOT)
    return _motion_model


def warmup() -> None:
    """Load templates and execute one inference before scored traffic."""
    _get_detector().warmup()
    _get_motion_model()


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Decode, detect, track, and answer one complete source frame."""
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            'Camera command from frame %s was ignored: %s',
            feedback.frame,
            feedback.reason,
        )

    try:
        image = decode_view(request.view)
        observations = detect(image, request)
        tracked = _tracker.update(request.sequence_id, request.frame, observations)
        annotations = [
            DroneFlybyPredictionDto(
                object_id=detection.object_id,
                bbox=list(detection.bbox),
                confidence=round(min(1.0, max(0.0, detection.confidence)), 4),
            )
            for detection in tracked
        ]
    except Exception:
        logger.exception('Prediction failed on frame %s', request.frame)
        annotations = []

    response = DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )
    validate_response(response)
    return response


def detect(image, request: DroneFlybyPredictRequestDto) -> List[Detection]:
    """Detect reference templates and return frame-global normalized boxes."""
    motion_model = _get_motion_model()
    expected_boxes = {
        object_id: source_bbox_to_global(
            predicted_bbox,
            request.original_width,
            request.original_height,
        )
        for object_id in motion_model.object_ids
        if (predicted_bbox := motion_model.predict(object_id, request.frame)) is not None
    }
    template_detections = []
    if request.view.resolution_level > 0:
        template_detections = _get_detector().detect(
            image,
            request.view.source_region_xyxy,
            request.original_width,
            request.original_height,
            expected_boxes=expected_boxes,
        )
    detections = [
        Detection(item.object_id, item.bbox, item.confidence)
        for item in template_detections
    ]
    best_by_class = {
        object_id: max(
            (detection for detection in detections if detection.object_id == object_id),
            key=lambda detection: detection.confidence,
        )
        for object_id in {detection.object_id for detection in detections}
    }
    for object_id in motion_model.object_ids:
        direct_detection = best_by_class.get(object_id)
        predicted_bbox = motion_model.predict(object_id, request.frame)
        if predicted_bbox is None:
            continue
        motion_bbox = source_bbox_to_global(
            predicted_bbox,
            request.original_width,
            request.original_height,
        )
        if (
            direct_detection is not None
            and direct_detection.confidence >= 0.75
            and intersection_over_union(direct_detection.bbox, motion_bbox) >= 0.3
        ):
            continue
        if direct_detection is not None:
            detections.remove(direct_detection)
        detections.append(
            Detection(
                object_id,
                motion_bbox,
                0.34,
            )
        )
    return detections


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
) -> Optional[RequestedViewDto]:
    """Follow a constraint-aware serpentine sweep toward detailed views."""
    constraints = request.camera_constraints
    current = request.view
    current_level = current.resolution_level

    if current_level == 0 and 1 in constraints.allowed_resolution_levels:
        bounds = constraints.bounds_for_level(1)
        if bounds is not None:
            return RequestedViewDto(
                resolution_level=1,
                center_x=int((bounds.minimum_center_x + bounds.maximum_center_x) // 2),
                center_y=int((bounds.minimum_center_y + bounds.maximum_center_y) // 2),
            )

    if current_level == 1 and 2 in constraints.allowed_resolution_levels:
        bounds = constraints.bounds_for_level(2)
        if bounds is not None:
            return RequestedViewDto(
                resolution_level=2,
                center_x=int(min(max(current.center_x, bounds.minimum_center_x), bounds.maximum_center_x)),
                center_y=int(min(max(current.center_y, bounds.minimum_center_y), bounds.maximum_center_y)),
            )

    bounds = constraints.bounds_for_level(current_level)
    if bounds is None or current_level == 0:
        return None

    direction = _sweep_direction.setdefault(request.sequence_id, 1)
    limit = max(1.0, constraints.maximum_center_delta)
    step = max(1, int(limit * 0.8))
    candidate_x = current.center_x + direction * step
    if candidate_x > bounds.maximum_center_x or candidate_x < bounds.minimum_center_x:
        direction *= -1
        _sweep_direction[request.sequence_id] = direction
        candidate_x = current.center_x + direction * step
    candidate_x = int(min(max(candidate_x, bounds.minimum_center_x), bounds.maximum_center_x))
    return RequestedViewDto(
        resolution_level=current_level,
        center_x=candidate_x,
        center_y=int(min(max(current.center_y, bounds.minimum_center_y), bounds.maximum_center_y)),
    )
