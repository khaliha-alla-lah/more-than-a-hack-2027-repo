"""Inspect external short spells before and after decoding."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from evaluation import possession_metrics as pm
from . import dfl
from .release import load_models
from .schema import FEATURE_NAMES, COLLECTIVE_NAMES
from .benchmark import TEST
from .decode import decode
from .experiments import gap_lengths
from .features import extract
from .learn import collective_matrix, degrade, matrix, predict


def load_audit_models(directory, legacy=False):
    """Legacy bundles are opt-in for historical diagnostics, never production inference."""
    directory = Path(directory)
    if not legacy or (directory / "manifest.json").exists():
        return load_models(directory)
    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    names = manifest["models"]
    collective = manifest.get("collective_status", False)
    penalties = manifest["penalties"]
    if (not isinstance(names, list) or len(names) not in (1, 2) or
            any(not isinstance(name, str) or not name or Path(name).name != name
                or name in (".", "..") for name in names) or len(set(names)) != len(names)):
        raise ValueError("Legacy audit requires one or two distinct model files inside the bundle")
    if type(collective) is not bool or (penalties is not None and (
            not isinstance(penalties, list) or len(penalties) != 2 or
            any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
                for value in penalties))):
        raise ValueError("Invalid legacy status setting or decoder penalties")
    import lightgbm as lgb

    models = [lgb.Booster(model_file=str(directory / name)) for name in names]
    widths = [len(FEATURE_NAMES)] if len(models) == 1 else [
        len(COLLECTIVE_NAMES if collective else FEATURE_NAMES), len(FEATURE_NAMES)]
    if any(model.num_feature() != width for model, width in zip(models, widths)):
        raise ValueError("Legacy model feature counts do not match the baseline schema")
    warnings.warn("Auditing a legacy bundle without verified source or model hashes; "
                  "this is not a verified production release.", RuntimeWarning, stacklevel=2)
    return models, penalties, collective


def audit_half(tracking, truth, models, penalties, collective):
    features = extract(tracking)
    probabilities = predict(models, matrix(features),
                            collective_matrix(features) if collective else None)
    raw = probabilities.argmax(axis=1)
    decoded = decode(probabilities, *penalties) if penalties else raw.copy()
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
    parser.add_argument("--legacy-model", action="store_true",
                        help="Explicitly audit old benchmark.json-only weights without provenance verification")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/short-spells.json"))
    args = parser.parse_args()
    model = load_audit_models(args.model_dir, args.legacy_model)
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
