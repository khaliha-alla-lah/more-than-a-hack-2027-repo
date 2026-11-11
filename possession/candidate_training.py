"""External-match training using the benchmarked recipe."""
import numpy as np
from .candidate import GBM, team_features, team_predict, turn_features
from .vendor.sam.gbm_model import build_turnover_targets, compute_spell_balanced_weights


def train(rows, rounds=(380, 420, 260)):
    dx, dy, tx, ty, tw, ux, uy, uw = ([] for _ in range(8))
    params = {"num_threads": 4, "verbosity": -1}
    for row, y in rows:
        dead = row[-1]
        idx = np.flatnonzero(y < 3)[::8]
        dx.append(dead["X_dead"][idx])
        dy.append(y[idx] == 2)
    dm = GBM.fit_dead_booster(dx, dy, dead["dead_feature_names"], n_rounds=rounds[0], params=params)
    team_inputs = []
    for row, y in rows:
        pd = GBM.predict_dead_with_booster(dm, row[-1]["X_dead"])
        tf = team_features(row, pd)
        letters = np.array(list("HADX"))[y]
        idx = np.flatnonzero(y < 2)[::8]
        tx.append(tf["X_team"][idx])
        ty.append(y[idx] == 0)
        tw.append(compute_spell_balanced_weights(letters, idx, row[3]))
        team_inputs.append((pd, tf, letters))
    tm = GBM.fit_team_booster(tx, ty, tw, tf["team_feature_names"],
                             tf["team_swap_pairs"], tf["team_negate_cols"],
                             n_rounds=rounds[1], params=params)
    for (row, y), (pd, tf, letters) in zip(rows, team_inputs, strict=True):
        ph = team_predict(tm, tf)
        uf = turn_features(row, pd, ph)
        target = build_turnover_targets(letters, row[3], radius_frames=3)
        idx = np.flatnonzero(y < 2)[::4]
        ux.append(uf["X_turn"][idx])
        uy.append(target[idx])
        uw.append(np.where(target[idx] > 0, 4., 1.))
    um = GBM.fit_turn_booster(ux, uy, uw, uf["turn_feature_names"], n_rounds=rounds[2], params=params)
    return dm, tm, um
