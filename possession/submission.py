"""Frozen submission inference; no training, labels, or research-directory imports."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

from games import GAME, GAMES, half_video_blocks
from evaluation.possession_metrics import read_label_csv, validate_submission
from .vendor.tracking.core.io import load_epts
from .vendor.tracking.core import features
from .vendor.tracking.core.decode import emissions, viterbi, LABELS

DEFAULT_MODELS = Path(__file__).resolve().parents[1] / 'models' / 'submission'


def extract_features(match):
    """Reset windows at each half and retain missing observations as NaN."""
    output = None
    for start, stop in match.halves:
        half = replace(match, x=match.x[start:stop], y=match.y[start:stop],
                       speed=match.speed[start:stop], ball_x=match.ball_x[start:stop],
                       ball_y=match.ball_y[start:stop], halves=[(0, stop-start)])
        computed = features.compute(half)
        if output is None:
            output = {'team': {k: {name: np.full(match.n_frames, np.nan, dtype=np.float32)
                                  for name in computed['team'][k]} for k in (0, 1)},
                      'shared': {name: np.full(match.n_frames, np.nan, dtype=np.float32)
                                 for name in computed['shared']}, 'fps': match.fps}
        for k in (0, 1):
            for name, values in computed['team'][k].items():
                output['team'][k][name][start:stop] = values
        for name, values in computed['shared'].items():
            output['shared'][name][start:stop] = values
    return output


def unreliable_ball(match, feats):
    bad = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        speed = np.hypot(np.diff(match.ball_x[start:stop]), np.diff(match.ball_y[start:stop])) * match.fps
        hit = np.zeros(stop-start, dtype=bool)
        hit[1:] |= speed > 40
        hit[:-1] |= speed > 40
        radius = max(1, round(.4 * match.fps))
        prefix = np.r_[0, np.cumsum(hit)]
        i = np.arange(stop-start)
        bad[start:stop] = prefix[np.minimum(stop-start, i+radius+1)] > prefix[np.maximum(0, i-radius)]
    return (np.nan_to_num(feats['shared']['ball_known'], nan=0) < .5) | bad


def predict(match, baseline, robust):
    feats = extract_features(match)
    base_dead, base_home = baseline.predict(feats)
    robust_dead, robust_home = robust.predict(feats)
    weight = .75 * unreliable_ball(match, feats)
    dead = (1-weight)*base_dead + weight*robust_dead
    home = (1-weight)*base_home + weight*robust_home
    scores = emissions(dead, home, .75, 2.)
    labels = np.full(match.n_frames, 'X')
    for start, stop in match.halves:
        path = viterbi(scores[start:stop], np.full(stop-start, 64*match.fps/25),
                       40*match.fps/25, 40*match.fps/25, max(1, round(match.fps)))
        labels[start:stop] = LABELS[path]
    # Published second-half setup precedes the supplied kickoff boundary.
    labels[110808:113742] = 'D'
    return labels


def load_models(directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    models = []
    for name in ('baseline.pkl', 'robust.pkl'):
        path = directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['models'][name]:
            raise ValueError(f'Model checksum mismatch: {name}')
        # Only load repository-shipped checkpoints after verifying their hashes.
        with path.open('rb') as handle:
            models.append(pickle.load(handle))
    return models, manifest


def export(data, models, output, threads=3):
    tactical = data/'tactical' if (data/'tactical').is_dir() else data
    with threadpool_limits(limits=threads):
        fitted, manifest = load_models(models)
        match = load_epts(tactical/'BAR-ATH_tactical_atd.xml', tactical/'BAR-ATH_tactical_atd.txt',
                          pitch=GAMES[GAME]['pitch'], n_frames=GAMES[GAME]['n_video_frames'])
        match.halves = [(a, b+1) for _, a, b in half_video_blocks(GAME)]
        labels = predict(match, *fitted)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name+'.tmp')
    try:
        with pending.open('w', newline='', encoding='utf-8') as handle:
            handle.write('frame,label\n')
            handle.writelines(f'{i},{label}\n' for i, label in enumerate(labels))
        error = validate_submission(read_label_csv(pending), GAMES[GAME]['n_video_frames'])
        if error:
            raise ValueError(error)
        checksum = hashlib.sha256(pending.read_bytes()).hexdigest()
        if checksum != manifest['submission_sha256']:
            raise ValueError(f'Frozen submission mismatch: {checksum}; previous output preserved')
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Dataset root or tactical folder')
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    checksum = export(args.data, args.models, args.output, args.threads)
    print(json.dumps({'file': str(args.output), 'rows': 199319, 'sha256': checksum}))


if __name__ == '__main__':
    main()
