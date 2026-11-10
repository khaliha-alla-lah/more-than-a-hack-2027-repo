"""Choose on one development match, then report results on two untouched games."""

import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks
from starter.baseline import smooth

from . import dfl
from .decode import decode, rules
from .features import extract
from .learn import collective_matrix, degrade, fit, gaps, matrix, predict

TRAIN = ("J03WMX", "J03WOH", "J03WPY", "J03WR9")
DEV = "J03WOY"
TEST = ("J03WN1", "J03WQQ")


def measurements(truth, predictions, visible):
    runs = pm.valid_runs(truth)
    spell = pm.spell_scores(truth, predictions, runs)
    midpoint = spell["per_iou"][0.5]
    expected = pm.possession_spells(truth, runs)
    actual = pm.possession_spells(predictions, runs)
    overlaps = pm.spell_overlaps(expected, actual)
    matched = pm.matched_gt_flags(overlaps, len(expected), 0.5)
    short = np.array([13 <= end - start <= 37 for start, end, _ in expected], dtype=bool)
    hidden_live = (truth < 2) & ~visible
    hidden_scored = (truth < 3) & ~visible
    transition = pm.transition_scores(pm.transitions(truth, runs),
                                      pm.transitions(predictions, runs), 25)
    return {
        "spell_f1": spell["mean_f1"],
        "macro_f1": pm.macro_f1(pm.scored_confusion(pm.confusion(truth, predictions))),
        "missed_at_0_5": midpoint["n_missed"],
        "extra_at_0_5": midpoint["n_spurious"],
        "short_recall_at_0_5": float(matched[short].mean()) if short.any() else None,
        "short_spells": int(short.sum()),
        "transition_f1_25_frames": transition["f1"],
        "boundary_mean_abs_error_frames": transition["mean_abs_latency"],
        "hidden_live_team_accuracy": float((truth[hidden_live] == predictions[hidden_live]).mean())
        if hidden_live.any() else None,
        "hidden_live_dead_accuracy": float(
            ((truth[hidden_scored] == 2) == (predictions[hidden_scored] == 2)).mean())
        if hidden_scored.any() else None,
    }


def nearest(features):
    home = np.nan_to_num(features["near_h"], nan=np.inf)
    away = np.nan_to_num(features["near_a"], nan=np.inf)
    codes = np.where(np.minimum(home, away) == np.inf, -1, np.where(home <= away, 0, 1))
    return pm.label_codes(smooth(codes, 50))


def prepare(directory, match, lengths, scenarios, references=False):
    out = {name: [] for name in scenarios}
    for half, (tracking, truth) in enumerate(dfl.load(directory, match)):
        seed = int(match[-2:], 36) * 10 + half
        for name in scenarios:
            observed = tracking if name == "clean" else degrade(
                tracking, lengths, seed + (10000 if name == "stress" else 0), name == "stress")
            features = extract(observed)
            baselines = (nearest(features), decode(rules(features))) if references else (None, None)
            out[name].append((matrix(features), truth, collective_matrix(features),
                              features["ball_seen"], *baselines))
    print(f"{match}: H/A/D/unknown {np.bincount(np.concatenate([half[1] for half in out[scenarios[0]]]), minlength=4).tolist()}",
          flush=True)
    return out


def score(halves, models=None, penalties=None, reference=None, collective=False):
    truth, prediction, visibility = [], [], []
    for x, y, motion, visible, nearest_codes, rule_codes in halves:
        if reference:
            codes = nearest_codes if reference == "nearest" else rule_codes
        else:
            probabilities = predict(models, x, motion if collective else None)
            codes = decode(probabilities, *penalties) if penalties else probabilities.argmax(axis=1)
        truth.extend((y, np.array([3], dtype=np.int8)))
        prediction.extend((codes, np.array([3], dtype=np.int8)))
        visibility.extend((visible, np.array([False])))
    return measurements(np.concatenate(truth), np.concatenate(prediction),
                        np.concatenate(visibility))


def fit_and_evaluate(directory, tactical, output):
    missing = np.concatenate([
        gaps(np.isfinite(tactical.ball[start:end + 1]).all(axis=1))
        for _, start, end in half_video_blocks(GAME)
    ])
    if not len(missing):
        raise ValueError("Need missing-ball lengths from the tactical tracking")

    x = {"clean": [], "mixed": []}
    y = {"clean": [], "mixed": []}
    motion = []
    for match in TRAIN:
        prepared = prepare(directory, match, missing, ("clean", "degraded"))
        for scenario, halves in prepared.items():
            for values, truth, collective, *_ in halves:
                step = 4
                x["mixed"].append(values[::step].copy())
                y["mixed"].append(truth[::step].copy())
                motion.append(collective[::step].copy())
                if scenario == "clean":
                    x["clean"].append(values[::step].copy())
                    y["clean"].append(truth[::step].copy())
    clean_x, clean_y = np.concatenate(x["clean"]), np.concatenate(y["clean"])
    mixed_x, mixed_y = np.concatenate(x["mixed"]), np.concatenate(y["mixed"])
    models = {
        "two_clean": fit(clean_x, clean_y),
        "two_mixed": fit(mixed_x, mixed_y),
        "joint_mixed": fit(mixed_x, mixed_y, joint=True),
        "two_collective": fit(mixed_x, mixed_y, status=np.concatenate(motion)),
    }
    dev = prepare(directory, DEV, missing, ("clean", "degraded", "stress"))
    candidates = {}
    for name, model in models.items():
        for penalties in (None, (5., 7.), (12., 16.), (25., 30.)):
            label = f"{name}:{penalties}"
            results = {scenario: score(halves, model, penalties, collective=name == "two_collective")
                       for scenario, halves in dev.items()}
            average = np.mean([results[scenario]["spell_f1"] for scenario in ("degraded", "stress")])
            candidates[label] = (float(average), name, penalties, results)
            print(f"development {label}: {average:.4f}", flush=True)
    # DFL keeps the observed ball almost stationary during D. A status model that
    # uses that signal predicted virtually no stoppages on the Metrica input.
    transferable = [key for key in candidates if candidates[key][1] == "two_collective"]
    selected = max(transferable, key=lambda key: candidates[key][0])
    _, name, penalties, _ = candidates[selected]
    print(f"Selected on {DEV}: {selected}", flush=True)
    held_out = {}
    for match in TEST:
        held_out[match] = {}
        prepared = prepare(directory, match, missing, ("clean", "degraded", "stress"), references=True)
        for scenario, halves in prepared.items():
            result = {
                "nearest": score(halves, reference="nearest"),
                "rules": score(halves, reference="rules"),
                "selected_per_frame": score(halves, models[name], collective=True),
                "selected": score(halves, models[name], penalties, collective=True),
            }
            held_out[match][scenario] = result
            print(match, scenario, {key: round(row["spell_f1"], 4) for key, row in result.items()},
                  flush=True)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_files = []
    for index, model in enumerate(models[name]):
        filename = f"{name}-{index}.txt"
        model.save_model(str(output / filename))
        model_files.append(filename)
    manifest = {"selected": selected, "penalties": penalties, "models": model_files,
                "collective_status": True,
                "train": TRAIN, "development": DEV, "test": TEST,
                "development_results": candidates, "held_out": held_out,
                "dfl_revision": dfl.REVISION}
    (output / "benchmark.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
