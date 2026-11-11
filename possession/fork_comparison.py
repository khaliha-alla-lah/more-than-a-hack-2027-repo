"""Run public fork components against shared inputs and the official scorer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import dfl
from .benchmark import measurements
from .corruption import corrupt
from .decode import decode, rules
from .experiments import SCENARIOS, PENALTIES, gap_lengths
from .features import extract, velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--blaze", type=Path, default=Path("data/fork-blaze"))
    parser.add_argument("--sam", type=Path, default=Path("data/fork-sam"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/fork-components.json"))
    args = parser.parse_args()
    sys.path[:0] = [str(args.blaze.resolve()), str(args.sam.resolve())]
    from solution.common import Match
    from solution.model import features, predict_hmm
    from possession_model.sequence_decoder import decode_possession_sequence

    lengths = gap_lengths(args.data)
    report = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for match in dfl.MATCHES:
        halves = list(dfl.load(args.dfl, match))
        match_report = {}
        for seed in (8001, 9001):
            for scenario in ("clean", "correlated", "stress"):
                results = {name: [] for name in ("rules", "blaze_rules", "rules_sam_decoder")}
                for half, (tracking, truth) in enumerate(halves):
                    observed = corrupt(tracking, lengths, seed + half, **SCENARIOS[scenario])
                    inputs = extract(observed)
                    probabilities = rules(inputs)
                    frames = len(truth)
                    shift = np.array(observed.pitch, dtype=np.float32) / 2
                    fork_match = Match(
                        match, observed.ball + shift, observed.players + shift,
                        np.linalg.norm(velocity(observed.players, observed.fps, 13.), axis=-1),
                        observed.teams, [(0, frames)], pitch=observed.pitch,
                    )
                    turnover, stoppage = PENALTIES
                    predictions = {
                        "rules": decode(probabilities, *PENALTIES),
                        "blaze_rules": predict_hmm(features(fork_match)),
                        "rules_sam_decoder": decode_possession_sequence(
                            -np.log(np.clip(probabilities, 1e-6, 1.)),
                            np.full(frames, turnover), np.full(frames, stoppage),
                            np.full(frames, stoppage),
                        ),
                    }
                    for name, prediction in predictions.items():
                        results[name].append((truth, prediction, inputs["ball_seen"]))
                match_report[f"{scenario}/{seed}"] = {
                    name: measurements(*(np.concatenate((rows[0][i], [False if i == 2 else 3],
                                                          rows[1][i])) for i in range(3)))
                    for name, rows in results.items()
                }
        report[match] = match_report
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Scored fork components for {match}", flush=True)


if __name__ == "__main__":
    main()
