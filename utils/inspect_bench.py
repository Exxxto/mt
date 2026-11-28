"""Inspect saved frames without ADB: python -m utils.inspect_bench PATH."""
import argparse
from pathlib import Path

import cv2

from core.bench_reader import BenchReader


def inspect_image(source, output, reader):
    source, output = Path(source), Path(output)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f'Cannot read image: {source}')
    if image.shape != (1600, 900, 3):
        raise ValueError(f'Expected 900x1600 image: {source}')
    observation = reader.read(image)
    canvas = image.copy()
    output.mkdir(parents=True, exist_ok=True)
    for slot in observation.slots:
        x0, y0, x1, y1 = slot.bounds
        color = {'occupied': (0, 255, 0), 'empty': (255, 200, 0), 'unknown': (0, 0, 255)}[slot.occupancy]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.circle(canvas, slot.center, 4, color, -1)
        bg_score = f'{slot.occupancy_score:.2f}' if slot.occupancy_score is not None else '?'
        match_score = f'{slot.recognition_score:.2f}' if slot.recognition_score is not None else '?'
        lines = [f'{slot.index} {slot.occupancy}', slot.unit_name or '?',
                 f'stars:{slot.stars or "?"} bg:{bg_score}', f'template:{match_score}']
        for row, line in enumerate(lines):
            cv2.putText(canvas, line[:19], (x0, 1080 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f'{source.stem}_slot_{slot.index}.png'), image[1180:y1, x0:x1])
    cv2.imwrite(str(output / f'{source.stem}_annotated.png'), canvas)
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path, default=Path('debug_images/bench_inspection'))
    args = parser.parse_args()
    files = sorted(args.path.glob('*.png')) if args.path.is_dir() else [args.path]
    reader = BenchReader()
    for source in files:
        try:
            inspect_image(source, args.output, reader)
        except ValueError as exc:
            print(exc)


if __name__ == '__main__':
    main()
