"""Generate copy-paste training scenes from the Helsinki reference boxes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectCrop:
    object_id: str
    image: np.ndarray


def load_reference_crops(
    image_directory: Path,
    annotation_directory: Path,
    padding: int = 4,
) -> Dict[str, ObjectCrop]:
    """Extract one representative crop for each reference object class."""
    crops: Dict[str, ObjectCrop] = {}
    for annotation_path in sorted(annotation_directory.glob('frame_*.json')):
        payload = json.loads(annotation_path.read_text())
        image_path = image_directory / f"frame_{int(payload['frame']):06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        for annotation in payload['annotations']:
            object_id = annotation['object_id']
            if object_id in crops:
                continue
            x1, y1, x2, y2 = (int(value) for value in annotation['bbox'])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size:
                crops[object_id] = ObjectCrop(object_id, crop)
    return crops


def load_reference_crop_sets(
    image_directory: Path,
    annotation_directory: Path,
    padding: int = 4,
) -> Dict[str, List[ObjectCrop]]:
    """Extract all available reference crops grouped by object class."""
    crop_sets: Dict[str, List[ObjectCrop]] = {}
    for annotation_path in sorted(annotation_directory.glob('frame_*.json')):
        payload = json.loads(annotation_path.read_text())
        image_path = image_directory / f"frame_{int(payload['frame']):06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        for annotation in payload['annotations']:
            x1, y1, x2, y2 = (int(value) for value in annotation['bbox'])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size:
                object_id = annotation['object_id']
                crop_sets.setdefault(object_id, []).append(ObjectCrop(object_id, crop))
    return crop_sets


def generate_scene(
    background: np.ndarray,
    object_crops: List[ObjectCrop],
    rng: np.random.Generator,
    view_scale: float = 1.0,
) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]:
    """Paste one random instance of each class onto a background image."""
    scene = background.copy()
    height, width = scene.shape[:2]
    labels: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for object_crop in object_crops:
        crop = object_crop.image
        scale = view_scale * float(rng.uniform(0.65, 1.35))
        new_width = max(4, int(crop.shape[1] * scale))
        new_height = max(4, int(crop.shape[0] * scale))
        resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            resized = cv2.flip(resized, 1)
        brightness = float(rng.uniform(0.75, 1.25))
        resized = np.clip(resized.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if new_width >= width or new_height >= height:
            continue
        x1 = int(rng.integers(0, width - new_width))
        y1 = int(rng.integers(0, height - new_height))
        x2, y2 = x1 + new_width, y1 + new_height
        scene[y1:y2, x1:x2] = resized
        labels.append((object_crop.object_id, (x1, y1, x2, y2)))
    return scene, labels


def write_dataset(
    project_root: Path,
    output_directory: Path,
    count: int = 100,
    seed: int = 7,
) -> None:
    """Write PNG scenes and class-indexed YOLO label files."""
    image_directory = project_root / 'src' / 'helsinki' / 'images'
    annotation_directory = project_root / 'src' / 'helsinki' / 'annotations'
    reference_crops = load_reference_crops(image_directory, annotation_directory)
    if len(reference_crops) != 16:
        raise ValueError(f'Expected 16 reference crops, found {len(reference_crops)}')

    output_directory.mkdir(parents=True, exist_ok=True)
    images_directory = output_directory / 'images'
    labels_directory = output_directory / 'labels'
    images_directory.mkdir(exist_ok=True)
    labels_directory.mkdir(exist_ok=True)
    class_names = sorted(reference_crops)
    (output_directory / 'classes.txt').write_text('\n'.join(class_names) + '\n')

    rng = np.random.default_rng(seed)
    background_paths = sorted(image_directory.glob('frame_*.png'))
    for index in range(count):
        background_path = background_paths[index % len(background_paths)]
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(background_path)
        background = cv2.resize(background, (960, 540), interpolation=cv2.INTER_AREA)
        view_scale = float(rng.choice((0.25, 0.5, 1.0)))
        scene, annotations = generate_scene(
            background,
            list(reference_crops.values()),
            rng,
            view_scale=view_scale,
        )
        image_name = f'scene_{index:06d}'
        cv2.imwrite(str(images_directory / f'{image_name}.png'), scene)
        height, width = scene.shape[:2]
        lines = []
        for object_id, (x1, y1, x2, y2) in annotations:
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f'{class_names.index(object_id)} {center_x:.6f} {center_y:.6f} '
                f'{box_width:.6f} {box_height:.6f}'
            )
        (labels_directory / f'{image_name}.txt').write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/synthetic'))
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    write_dataset(project_root, args.output, args.count, args.seed)
    print(f'generated {args.count} scenes in {args.output}')


if __name__ == '__main__':
    main()
