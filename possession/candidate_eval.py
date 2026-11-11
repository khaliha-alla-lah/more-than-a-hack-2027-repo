"""Evaluate the candidate with complete matches held out from training."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import dfl
from .benchmark import measurements
from .candidate import prepare, predict
from .candidate_training import train
from .corruption import corrupt
from .experiments import SCENARIOS, gap_lengths
from evaluation import possession_metrics as pm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--matches", nargs="+", choices=dfl.MATCHES, default=dfl.MATCHES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = gap_lengths(args.data)
    raw = {match: list(dfl.load(args.dfl, match)) for match in dfl.MATCHES}
    cache, report = {}, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for held_out in args.matches:
        training = []
        for index, (match, halves) in enumerate(raw.items()):
            if match == held_out:
                continue
            for half, (tracking, truth) in enumerate(halves):
                key = match, half
                if key not in cache:
                    observed = corrupt(tracking, lengths, 417+index*10+half,
                                       **SCENARIOS["missing"])
                    cache[key] = prepare(observed), truth
                training.append(cache[key])
        models = train(training)
        result = {}
        for seed in (8001, 9001):
            for scenario in ("missing", "correlated", "stress"):
                truths, predictions, visibility = [], [], []
                for half, (tracking, truth) in enumerate(raw[held_out]):
                    observed = corrupt(tracking, lengths, seed+half, **SCENARIOS[scenario])
                    labels, _, _ = predict(observed, models)
                    truths.extend((truth, [3]))
                    predictions.extend((pm.label_codes(labels), [3]))
                    visibility.extend((np.isfinite(observed.ball).all(axis=1), [False]))
                result[f"{scenario}/{seed}"] = measurements(
                    np.concatenate(truths), np.concatenate(predictions), np.concatenate(visibility))
        report[held_out] = result
        pending = args.output.with_suffix(".pending.json")
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(args.output)
        print(f"Evaluated {held_out}", flush=True)


if __name__ == "__main__":
    main()
