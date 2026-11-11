"""Compare augmentation and status models on the same whole-match folds."""

import argparse
import json
from pathlib import Path

import numpy as np

from games import GAME, half_video_blocks
from . import dfl
from .benchmark import nearest, score
from .corruption import corrupt
from .decode import decode, rules
from .features import extract
from .learn import collective_matrix, fit, gaps, matrix
from .tracking import load_metrica


SCENARIOS = {
    "clean": {},
    "ball_only": {"ball_loss": True},
    "players_only": {"player_loss": True},
    "jitter_only": {"jitter": True, "rho": 0},
    "missing": {"ball_loss": True, "player_loss": True},
    "independent": {"ball_loss": True, "player_loss": True, "jitter": True, "rho": 0},
    "correlated": {"ball_loss": True, "player_loss": True, "jitter": True},
    "stress": {"ball_loss": True, "player_loss": True, "jitter": True, "stress": True},
}
TRAINING = ("missing", "independent", "correlated")
PENALTIES = (12., 16.)


def prepared(directory, match, lengths, scenarios, seed):
    output = {scenario: [] for scenario in scenarios}
    for half, (tracking, truth) in enumerate(dfl.load(directory, match)):
        for scenario in scenarios:
            observed = corrupt(tracking, lengths, seed + half, **SCENARIOS[scenario])
            features = extract(observed)
            output[scenario].append((
                matrix(features), truth, collective_matrix(features), features["ball_seen"],
                nearest(features), decode(rules(features)),
            ))
    return output


def gap_lengths(data):
    tracking = load_metrica(data)
    return np.concatenate([
        gaps(np.isfinite(tracking.ball[start:stop + 1]).all(axis=1))
        for _, start, stop in half_video_blocks(GAME)
    ])


def training_rows(directory, lengths, matches, scenarios):
    training = {}
    for index, match in enumerate(matches):
        prepared_match = prepared(directory, match, lengths, ("clean", *scenarios), 417 + index * 10)
        training[match] = {
            scenario: tuple(np.concatenate([half[column][::4] for half in halves])
                            for column in range(3))
            for scenario, halves in prepared_match.items()
        }
        del prepared_match
        print(f"Prepared {match}", flush=True)
    return training


def fit_candidate(directory, data, model_dir, matches, scenario, collective):
    training = training_rows(directory, gap_lengths(data), matches, (scenario,))
    rows = [training[match][condition] for match in matches for condition in ("clean", scenario)]
    x, y, motion = (np.concatenate([row[column] for row in rows]) for column in range(3))
    models = fit(x, y, status=motion if collective else None)
    model_dir.mkdir(parents=True, exist_ok=True)
    names = ("live.txt", "team.txt")
    for model, name in zip(models, names, strict=True):
        model.save_model(str(model_dir / name))
    manifest = {"models": names, "penalties": PENALTIES, "collective_status": collective,
                "train": matches, "corruption": SCENARIOS[scenario],
                "dfl_revision": dfl.REVISION}
    (model_dir / "benchmark.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(directory, data, output, matches, seeds):
    lengths = gap_lengths(data)
    training = training_rows(directory, lengths, matches, TRAINING)

    report = {"folds": {}, "seeds": seeds, "penalties": PENALTIES,
              "scenarios": SCENARIOS, "dfl_revision": dfl.REVISION}
    output.parent.mkdir(parents=True, exist_ok=True)
    for match in matches:
        models = {}
        for scenario in TRAINING:
            rows = [training[other][condition] for other in matches if other != match
                    for condition in ("clean", scenario)]
            x, y, motion = (np.concatenate([row[column] for row in rows]) for column in range(3))
            models[f"{scenario}/collective"] = (fit(x, y, status=motion), True)
            models[f"{scenario}/full"] = (fit(x, y), False)
        fold = {}
        for seed in seeds:
            evaluation = prepared(directory, match, lengths, SCENARIOS, seed)
            fold[str(seed)] = {
                scenario: {
                    "nearest": score(halves, reference="nearest"),
                    "rules": score(halves, reference="rules"),
                    **{name: score(halves, model, PENALTIES, collective=collective)
                       for name, (model, collective) in models.items()},
                }
                for scenario, halves in evaluation.items()
            }
            del evaluation
        report["folds"][match] = fold
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Scored {match}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/benchmark.json"))
    parser.add_argument("--matches", nargs="+", default=dfl.MATCHES)
    parser.add_argument("--seeds", nargs="+", type=int, default=(8001, 9001))
    parser.add_argument("--fit-candidate", choices=TRAINING)
    parser.add_argument("--collective-status", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("data/experiments/candidate"))
    args = parser.parse_args()
    if len(set(args.matches)) != len(args.matches) or len(args.matches) < 2:
        parser.error("Provide at least two distinct matches")
    if args.fit_candidate:
        fit_candidate(args.dfl, args.data, args.model_dir, args.matches, args.fit_candidate,
                      args.collective_status)
    else:
        run(args.dfl, args.data, args.output, args.matches, args.seeds)


if __name__ == "__main__":
    main()
