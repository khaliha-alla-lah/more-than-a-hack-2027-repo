"""Model training and input degradation for external matches."""

import numpy as np

from .decode import combine
from .tracking import Tracking


def matrix(features):
    return np.ascontiguousarray(np.column_stack(list(features.values())), dtype=np.float32)


def collective_matrix(features):
    names = ("movement_h", "movement_a", "centroid_speed_h", "centroid_speed_a",
             "spread_h", "spread_a")
    return matrix({name: features[name] for name in names})


def gaps(observed):
    edges = np.flatnonzero(np.diff(np.r_[False, ~observed, False]))
    return edges[1::2] - edges[::2]


def degrade(tracking, missing_lengths, seed, stress=False):
    """Sample missing bursts without using possession or stoppage labels."""
    rng = np.random.default_rng(seed)
    players = tracking.players.copy()
    ball = tracking.ball.copy()
    lengths = np.asarray(missing_lengths)
    mask = np.zeros(len(ball), dtype=bool)
    fraction = 0.5 if stress else 0.4
    while mask.mean() < fraction:
        length = min(len(ball), int(rng.choice(lengths)))
        start = rng.integers(len(ball) - length + 1)
        mask[start:start + length] = True
    ball[mask] = np.nan
    players += rng.normal(0., 0.14 if stress else 0.08, players.shape).astype(np.float32)
    ball += rng.normal(0., 0.28 if stress else 0.15, ball.shape).astype(np.float32)
    for slot in range(players.shape[1]):
        for start in rng.integers(len(players), size=max(1, len(players) // 1500)):
            players[start:start + rng.integers(30, 180), slot] = np.nan
    return Tracking(players, tracking.teams, ball, tracking.pitch, tracking.fps)


def fit(x, y, joint=False, status=None):
    import lightgbm as lgb

    params = dict(verbosity=-1, learning_rate=0.08, num_leaves=23,
                  min_data_in_leaf=120, force_col_wise=True, deterministic=True,
                  num_threads=4, seed=417)
    known = y < 3
    if joint:
        dataset = lgb.Dataset(x[known], label=y[known])
        return (lgb.train(dict(params, objective="multiclass", num_class=3), dataset,
                          num_boost_round=100),)
    live = y < 2
    status_input = x if status is None else status
    status_data = lgb.Dataset(status_input[known], label=live[known])
    control_data = lgb.Dataset(x[live], label=y[live] == 0)
    params["objective"] = "binary"
    return (lgb.train(params, status_data, num_boost_round=100),
            lgb.train(params, control_data, num_boost_round=100))


def predict(models, x, status=None):
    if len(models) == 1:
        return models[0].predict(x, num_threads=4)
    status_input = x if status is None else status
    return combine(models[0].predict(status_input, num_threads=4),
                   models[1].predict(x, num_threads=4))
