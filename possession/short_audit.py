"""Inspect external short spells before and after decoding."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from . import dfl
from .__main__ import load_models
from .benchmark import TEST
from .decode import decode
from .experiments import gap_lengths
from .features import extract
from .learn import collective_matrix, degrade, matrix, predict


def audit_half(tracking, truth, models, penalties, collective):
    features = extract(tracking)
    probabilities = predict(models, matrix(features),
                            collective_matrix(features) if collective else None)
    raw = probabilities.argmax(axis=1)
    decoded = decode(probabilities, *penalties)
    runs = pm.valid_runs(truth)
    spells = pm.possession_spells(truth, runs)
    stages = {}
    for name, labels in (("raw", raw), ("decoded", decoded)):
        predicted = pm.possession_spells(labels, runs)
        overlaps = pm.spell_overlaps(spells, predicted)
        best = np.zeros(len(spells))
        for iou, expected, _ in overlaps:
            best[expected] = max(best[expected], iou)
        stages[name] = best, pm.matched_gt_flags(overlaps, len(spells), .5)
    rows = []
    for index, (start, end, team) in enumerate(spells):
        if 13 <= end - start <= 37:
            rows.append({
                "start": start, "end": end, "team": int(team),
                "visible_fraction": float(features["ball_seen"][start:end].mean()),
                "mean_team_probability": float(probabilities[start:end, team].mean()),
                "raw_team_fraction": float((raw[start:end] == team).mean()),
                "decoded_team_fraction": float((decoded[start:end] == team).mean()),
                **{f"{name}_{key}": conversion(values[position][index])
                   for name, values in stages.items()
                   for position, (key, conversion) in enumerate(
                       (("best_iou", float), ("matched", bool)))},
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/frozen-candidate"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/short-spells.json"))
    args = parser.parse_args()
    model = load_models(args.model_dir)
    lengths = gap_lengths(args.data)
    output = {}
    for match in TEST:
        output[match] = []
        for half, (tracking, truth) in enumerate(dfl.load(args.dfl, match)):
            observed = degrade(tracking, lengths, int(match[-2:], 36) * 10 + half)
            output[match].append(audit_half(observed, truth, *model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
