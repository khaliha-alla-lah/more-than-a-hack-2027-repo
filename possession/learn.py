"""Model training and input degradation for external matches."""

import numpy as np

from .decode import combine
from .schema import COLLECTIVE_NAMES, FEATURE_NAMES
from .tracking import Tracking

TRAINING_PARAMS = dict(verbosity=-1, learning_rate=0.08, num_leaves=23,
                       min_data_in_leaf=120, force_col_wise=True, deterministic=True,
                       num_threads=4, seed=417)
NUM_BOOST_ROUND = 100


def matrix(features, names=FEATURE_NAMES):
    missing = set(names) - features.keys()
    extra = features.keys() - set(names)
    if missing or extra:
        raise ValueError(f"Feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return np.ascontiguousarray(np.column_stack([features[name] for name in names]),
                                dtype=np.float32)


def collective_matrix(features):
    return matrix({name: features[name] for name in COLLECTIVE_NAMES}, COLLECTIVE_NAMES)


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

    params = TRAINING_PARAMS.copy()
    known = y < 3
    if joint:
        dataset = lgb.Dataset(x[known], label=y[known])
        return (lgb.train(dict(params, objective="multiclass", num_class=3), dataset,
                          num_boost_round=NUM_BOOST_ROUND),)
    live = y < 2
    status_input = x if status is None else status
    status_data = lgb.Dataset(status_input[known], label=live[known])
    control_data = lgb.Dataset(x[live], label=y[live] == 0)
    params["objective"] = "binary"
    return (lgb.train(params, status_data, num_boost_round=NUM_BOOST_ROUND),
            lgb.train(params, control_data, num_boost_round=NUM_BOOST_ROUND))


def predict(models, x, status=None):
    if len(models) == 1:
        return models[0].predict(x, num_threads=4)
    status_input = x if status is None else status
    return combine(models[0].predict(status_input, num_threads=4),
                   models[1].predict(x, num_threads=4))
