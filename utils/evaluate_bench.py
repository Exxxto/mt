"""Compare the current bench reader with a Git revision on manually labelled frames.

Run as a module from the repository root; recordings remain local and untracked.
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import cv2

from core.bench_reader import BenchReader


def summarize(reader, frames, session):
    counts = collections.Counter()
    durations = []
    for item in frames:
        path = session / 'frames' / f"{item['frame_id']:06}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        observation = reader.read(image, item['frame_id'])
        durations.append(observation.duration_ms)
        for truth, prediction in zip(item['slots'], observation.slots):
            occupied = truth['occupancy'] == 'occupied'
            counts['slots'] += 1
            counts['occupied'] += occupied
            counts['unknown'] += prediction.occupancy == 'unknown'
            counts['tp'] += occupied and prediction.occupancy == 'occupied'
            counts['fp'] += not occupied and prediction.occupancy == 'occupied'
            counts['fn_empty'] += occupied and prediction.occupancy == 'empty'
            counts['tn'] += not occupied and prediction.occupancy == 'empty'
            if occupied and truth['unit_name'] is None:
                counts['identity_unlabelled'] += 1
                continue
            counts['identity_labeled_occupied'] += occupied
            if prediction.unit_name:
                counts['named'] += 1
                counts['correct_name'] += prediction.unit_name == truth['unit_name']
                counts['wrong_name'] += prediction.unit_name != truth['unit_name']
    return {'counts': dict(counts), 'mean_ms': statistics.mean(durations),
            'max_ms': max(durations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--labels', type=Path, default=Path('tests/fixtures/bench_stage4_holdout.json'))
    parser.add_argument('--baseline', default='e463f656bf75dd150a27bd64f12731e723e85483')
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding='utf8'))
    if not args.session.name.endswith(labels['session_id']):
        parser.error('session directory does not match the labels')
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        source = temp / 'baseline.py'
        source.write_bytes(subprocess.check_output(['git', 'show', f'{args.baseline}:core/bench_reader.py']))
        background = temp / 'empty_bench.png'
        background.write_bytes(subprocess.check_output(
            ['git', 'show', f'{args.baseline}:templates/empty_bench.png']))
        models = temp / 'models'
        models.mkdir()
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.baseline,
                                         'templates/bench']).decode('utf8').splitlines()
        for name in paths:
            if name.endswith('.png'):
                (models / Path(name).name).write_bytes(subprocess.check_output(
                    ['git', 'show', f'{args.baseline}:{name}']))
        spec = importlib.util.spec_from_file_location('bench_baseline', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name, reader in (('baseline', module.BenchReader(background_path=background, model_dir=models)),
                             ('current', BenchReader())):
            print(name, json.dumps(summarize(reader, labels['frames'], args.session), ensure_ascii=False))


if __name__ == '__main__':
    main()
