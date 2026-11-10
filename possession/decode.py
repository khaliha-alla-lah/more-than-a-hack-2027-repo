"""Sequence decoding over home, away, and dead ball."""

import numpy as np


def combine(live, home):
    return np.column_stack((live * home, live * (1. - home), 1. - live))


def decode(probabilities, turnover=5., stoppage=7.):
    scores = np.log(np.clip(probabilities, 1e-6, 1.))
    if not len(scores):
        return np.empty(0, dtype=np.int8)
    previous = np.zeros(scores.shape, dtype=np.int8)
    home, away, dead = scores[0]
    for index, (h, a, d) in enumerate(scores[1:], start=1):
        into_home = (home, away - turnover, dead - stoppage)
        into_away = (home - turnover, away, dead - stoppage)
        into_dead = (home - stoppage, away - stoppage, dead)
        best_home = max(range(3), key=into_home.__getitem__)
        best_away = max(range(3), key=into_away.__getitem__)
        best_dead = max(range(3), key=into_dead.__getitem__)
        previous[index] = (best_home, best_away, best_dead)
        home = into_home[best_home] + h
        away = into_away[best_away] + a
        dead = into_dead[best_dead] + d
    output = np.empty(len(scores), dtype=np.int8)
    output[-1] = np.argmax([home, away, dead])
    for index in range(len(output) - 1, 0, -1):
        output[index - 1] = previous[index, output[index]]
    return output


def rules(features):
    """Conservative estimates from control, player motion and pitch boundaries."""
    movement = np.fmax(features["movement_h"], features["movement_a"])
    motion = np.nan_to_num(movement, nan=1.4)
    edge = features["pitch_edge"]
    speed = np.nan_to_num(features["ball_speed_context"], nan=0.)
    # A quiet tracked ball can still be live. Motion and pitch position corroborate it.
    live_score = 2.0 * (motion - 1.7) + 0.22 * np.minimum(speed, 10.)
    live_score += np.where(features["ball_seen"], 0.25, 0.)
    live_score -= 2.2 * np.clip(np.nan_to_num(edge, nan=0.), 0., 2.)
    live = 1. / (1. + np.exp(-np.clip(live_score, -7., 7.)))

    current = np.clip(features["control_margin"], -5., 5.)
    context = np.nan_to_num(features["control_context"])
    last = np.nan_to_num(np.clip(features["last_control"], -4., 4.))
    following = np.nan_to_num(np.clip(features["next_control"], -4., 4.))
    since = np.nan_to_num(features["since_ball"], nan=120.)
    until = np.nan_to_num(features["until_ball"], nan=120.)
    gap_evidence = 0.65 * last * np.exp(-since / 5.) + 0.5 * following * np.exp(-until / 5.)
    team_motion = np.nan_to_num(features["movement_h"] - features["movement_a"])
    control = np.where(np.isfinite(current), 0.55 * current + 0.65 * context,
                       gap_evidence + 0.3 * team_motion)
    home = 1. / (1. + np.exp(-np.clip(control, -7., 7.)))
    return combine(live, home)
