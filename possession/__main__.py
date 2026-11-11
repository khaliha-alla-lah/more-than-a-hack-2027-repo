"""Generate a validated possession CSV from a frozen release or the rules baseline."""

import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks, require
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_tags

from .decode import decode, rules
from .features import extract
from .learn import collective_matrix, matrix, predict
from .release import load_models, run_receipt, write_json
from .tracking import load_metrica


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
        # Explicit UTF-8/LF makes the file contract independent of the host OS.
        with pending.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("frame,label\n")
            stream.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
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
    mode.add_argument("--fit", action="store_true", help="Train then infer (prefer python -m possession.train)")
    mode.add_argument("--rules", action="store_true", help="Generate without a learned model")
    parser.add_argument("--data", type=Path, help="Directory containing tactical/ and broadcast/")
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/possession-model"))
    parser.add_argument("--output", type=Path, default=Path("data/submission.csv"))
    parser.add_argument("--diagnostics", type=Path, help="Local CSV of uncertain transitions")
    args = parser.parse_args()

    try:
        if args.fit:
            from .train import train

            train(args.data, args.dfl, args.model_dir)
        # Fail before loading the large tracking file if the release is missing or invalid.
        models, penalties, collective = (None, (5., 7.), False) if args.rules else load_models(args.model_dir)
        input_paths = require(["atd_meta", "atd_raw"], data_dir=args.data)
        tracking = load_metrica(args.data)
        tags = []
        if args.diagnostics:
            if args.diagnostics.resolve() == args.output.resolve():
                raise ValueError("Diagnostics and submission must use different paths")
            tags = [tag for tag in load_tags(data_dir=args.data) if tag["code"] == "SET PIECES"]
        generate(tracking, args.output, models, penalties, collective, args.diagnostics, tags)
        receipt = run_receipt(args.output, dict(zip(("tactical_metadata", "tactical_tracking"), input_paths)),
                              mode="rules" if args.rules else "learned", penalties=penalties,
                              model_dir=None if args.rules else args.model_dir, rows=len(tracking.ball))
        receipt_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        write_json(receipt_path, receipt)
        print(f"SHA-256 {receipt['output']['sha256']}\nRun receipt: {receipt_path}", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Possession error: {error}\n")


if __name__ == "__main__":
    main()
