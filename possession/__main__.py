"""Run the rules entry or train and run the selected external-data model."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks
from starter.baseline import write_submission
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_tags

from .decode import decode, rules
from .features import extract
from .learn import collective_matrix, matrix, predict
from .tracking import load_metrica


def load_models(directory):
    import lightgbm as lgb

    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    models = [lgb.Booster(model_file=str(directory / name)) for name in manifest["models"]]
    return models, manifest["penalties"], manifest.get("collective_status", False)


def diagnostics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "seconds", "prediction", "ball_visible",
                         "p_home", "p_away", "p_dead", "nearest_home_m", "nearest_away_m",
                         "home_motion_m_s", "away_motion_m_s", "set_piece_window"))
        writer.writerows(rows)


def generate(tracking, output, models=None, penalties=(5., 7.), collective=False,
             diagnostic_path=None, tags=()):
    labels = np.full(len(tracking.ball), "X", dtype="<U1")
    kickoff = KICKOFF_VIDEO_FRAMES["2nd Half"]
    rows = []
    for half, start, stop in half_video_blocks(GAME):
        end = stop + 1
        if half == 2:
            # The official FAQ identifies this graded pre-kickoff span as D.
            labels[start:kickoff] = "D"
            start = kickoff
        features = extract(tracking.section(start, end))
        status = collective_matrix(features) if collective else None
        probabilities = predict(models, matrix(features), status) if models else rules(features)
        codes = decode(probabilities, *penalties) if penalties else probabilities.argmax(axis=1)
        labels[start:end] = np.asarray(pm.LABELS)[codes]
        if diagnostic_path:
            # Inspect transitions with weak frame-level evidence locally.
            changes = np.flatnonzero(np.r_[False, codes[1:] != codes[:-1]])
            uncertain = changes[np.max(probabilities[changes], axis=1) < 0.8]
            for index in uncertain:
                frame = start + int(index)
                restart = any(tag["start_frame"] <= frame < tag["end_frame"] for tag in tags)
                rows.append((frame, frame / tracking.fps, labels[frame],
                             int(features["ball_seen"][index]),
                             *np.round(probabilities[index], 5),
                             *(round(float(features[name][index]), 2) for name in
                               ("near_h", "near_a", "movement_h", "movement_a")),
                             int(restart)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".csv", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        write_submission(str(pending), labels)
        problem = pm.validate_submission(pm.read_label_csv(pending), len(labels))
        if problem:
            raise ValueError(problem)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    if diagnostic_path:
        diagnostics(diagnostic_path, rows)
    print(f"Validated {len(labels):,} rows in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit", action="store_true", help="Train, evaluate and generate")
    mode.add_argument("--rules", action="store_true", help="Generate without a learned model")
    parser.add_argument("--data", type=Path, help="Directory containing tactical/ and broadcast/")
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/possession-model"))
    parser.add_argument("--output", type=Path, default=Path("data/submission.csv"))
    parser.add_argument("--diagnostics", type=Path, help="Local CSV of uncertain transitions")
    args = parser.parse_args()

    tracking = load_metrica(args.data)
    if args.fit:
        from .benchmark import fit_and_evaluate

        fit_and_evaluate(args.dfl, tracking, args.model_dir)
    models, penalties, collective = (None, (5., 7.), False) if args.rules else load_models(args.model_dir)
    tags = []
    if args.diagnostics:
        tags = [tag for tag in load_tags(data_dir=args.data) if tag["code"] == "SET PIECES"]
    generate(tracking, args.output, models, penalties, collective, args.diagnostics, tags)


if __name__ == "__main__":
    main()
