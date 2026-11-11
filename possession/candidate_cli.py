"""Train or run the motion candidate, with local diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import lightgbm as lgb
import numpy as np

from games import GAME, half_video_blocks
from starter.baseline import write_submission
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
from evaluation import possession_metrics as pm
from . import dfl
from .candidate import prepare, predict
from .candidate_training import train
from .corruption import corrupt
from .learn import gaps
from .tracking import load_metrica


def fit(directory, tracking, model_dir):
    lengths = np.concatenate([gaps(np.isfinite(tracking.ball[s:e+1]).all(axis=1))
                              for _, s, e in half_video_blocks(GAME)])
    rows = []
    for index, match in enumerate(dfl.MATCHES):
        for half, (observed, truth) in enumerate(dfl.load(directory, match)):
            noisy = corrupt(observed, lengths, 417+index*10+half, ball_loss=True, player_loss=True)
            rows.append((prepare(noisy), truth))
        print(f"Prepared {match}", flush=True)
    models = train(rows)
    model_dir.mkdir(parents=True, exist_ok=True)
    version = uuid4().hex
    files = []
    for name, model in zip(("dead", "team", "turn"), models, strict=True):
        path = model_dir / f"{version}-{name}.txt"
        model.save_model(str(path))
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"format": 1, "models": files, "train": dfl.MATCHES,
                "rounds": [380,420,260], "velocity_window": 7, "dfl_revision": dfl.REVISION,
                "source_revision": "3b800fb11184a57dc84baab819c21142fbc57513"}
    pending = model_dir / f"{version}-manifest.json"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending.replace(model_dir / "manifest.json")
    return models


def load(model_dir):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest["format"] != 1 or manifest["velocity_window"] != 7:
        raise ValueError("Unsupported candidate model format")
    models = []
    for record in manifest["models"]:
        path = model_dir / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
        models.append(lgb.Booster(model_file=str(path)))
    if len(models) != 3:
        raise ValueError("Candidate requires three models")
    return models


def generate(tracking, models, output, diagnostics=None):
    labels = np.full(len(tracking.ball), "X", dtype="<U1")
    records = []
    for half, start, stop in half_video_blocks(GAME):
        if half == 2:
            kickoff = KICKOFF_VIDEO_FRAMES["2nd Half"]
            labels[start:kickoff] = "D"
            start = kickoff
        section = tracking.section(start, stop+1)
        predicted, probabilities, turns = predict(section, models)
        labels[start:stop+1] = predicted
        if diagnostics:
            changes = np.flatnonzero(np.r_[False, predicted[1:] != predicted[:-1]])
            regular = np.arange(0, len(predicted), 250)
            disagreement = np.flatnonzero(
                predicted != np.array(list("HAD"))[probabilities.argmax(axis=1)])[::25]
            for index in np.unique(np.concatenate((changes, regular, disagreement))):
                records.append((start+index, (start+index)/tracking.fps, predicted[index],
                                *probabilities[index].tolist(), float(turns[index]),
                                int(np.isfinite(section.ball[index]).all())))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".csv", delete=False) as handle:
        pending = Path(handle.name)
    try:
        write_submission(str(pending), labels)
        error = pm.validate_submission(pm.read_label_csv(pending), len(labels))
        if error:
            raise ValueError(error)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    if diagnostics:
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame","seconds","prediction","p_home","p_away","p_dead",
                             "p_turnover","ball_visible"))
            writer.writerows(records)
    print(f"Validated {len(labels):,} rows in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/motion-candidate"))
    parser.add_argument("--output", type=Path, default=Path("data/motion-candidate/submission.csv"))
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    tracking = load_metrica(args.data)
    models = fit(args.dfl, tracking, args.model_dir) if args.fit else load(args.model_dir)
    generate(tracking, models, args.output, args.diagnostics)


if __name__ == "__main__":
    main()
