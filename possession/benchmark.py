"""Reconstruct the historical split; reporting games have already informed development."""

import hashlib

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks
from starter.baseline import smooth

from . import dfl
from .decode import decode, rules
from .features import extract
from .learn import (NUM_BOOST_ROUND, TRAINING_PARAMS, collective_matrix,
                    degrade, fit, gaps, matrix, predict)
from .release import check_destination, fingerprint, save_release

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


def fit_and_evaluate(directory, tactical, output, tactical_inputs=None):
    check_destination(output)
    # Fail on all missing files before starting an expensive partial reconstruction.
    inputs = [path for match in dfl.MATCHES for path in dfl.paths(directory, match)]
    missing_files = [str(path) for path in inputs if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing DFL inputs:\n" + "\n".join(missing_files))
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

    manifest = {"selected": selected, "penalties": penalties,
                "collective_status": True,
                "train": TRAIN, "development": DEV, "test": TEST,
                "test_status": "Previously inspected reporting games; not a pristine holdout",
                "development_results": candidates, "held_out": held_out,
                "dfl_revision": dfl.REVISION}
    provenance = {
        "purpose": "Reconstruction of the historical baseline; not recovered original weights",
        "dfl_revision": dfl.REVISION,
        "dfl_inputs": {path.name: fingerprint(path) for path in inputs},
        "tactical_inputs": {name: fingerprint(path) for name, path in (tactical_inputs or {}).items()},
        "split": {"train": TRAIN, "development": DEV, "reporting": TEST},
        "lightgbm_params": TRAINING_PARAMS, "num_boost_round": NUM_BOOST_ROUND,
        "training_stride": 4,
        "augmentation": {"recipe": "historical independent position jitter, unchanged",
                         "seed": "int(match[-2:], 36) * 10 + zero_based_half; stress adds 10000",
                         "missing_fraction": {"degraded": 0.4, "stress": 0.5},
                         "player_noise_std_m": {"degraded": 0.08, "stress": 0.14},
                         "ball_noise_std_m": {"degraded": 0.15, "stress": 0.28},
                         "missing_lengths_sha256": hashlib.sha256(missing.astype("<i8").tobytes()).hexdigest()},
        "selection": {"family": "two_collective", "metric": "mean spellF1 on degraded/stress development",
                      "penalties": [None, [5., 7.], [12., 16.], [25., 30.]]},
    }
    save_release(output, models[name], manifest, provenance)
    return manifest
