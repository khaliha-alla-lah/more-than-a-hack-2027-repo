"""Shared tracking features. No provider's answer fields enter this module."""

import numpy as np


def mean(values, axis):
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(count.shape, np.nan, dtype=np.float32), where=count > 0)


def velocity(points, fps, limit):
    result = np.full_like(points, np.nan)
    result[1:] = np.diff(points, axis=0) * fps
    result[np.linalg.norm(result, axis=-1) > limit] = np.nan
    return result


def window_mean(values, radius):
    index = np.arange(len(values))
    lo = np.maximum(0, index - radius)
    hi = np.minimum(len(values), index + radius + 1)
    finite = np.isfinite(values)
    total = np.r_[0., np.cumsum(np.where(finite, values, 0.), dtype=np.float64)]
    count = np.r_[0, np.cumsum(finite)]
    size = count[hi] - count[lo]
    return np.divide(total[hi] - total[lo], size,
                     out=np.full(len(values), np.nan), where=size > 0)


def extract(tracking):
    players, ball, fps = tracking.players, tracking.ball, tracking.fps
    observed = np.isfinite(ball).all(axis=1)
    index = np.arange(len(ball))
    last = np.maximum.accumulate(np.where(observed, index, -1))
    following = np.minimum.accumulate(np.where(observed, index, len(ball))[::-1])[::-1]
    previous_distance = np.where(last >= 0, (index - last) / fps, np.nan)
    next_distance = np.where(following < len(ball), (following - index) / fps, np.nan)
    ball_velocity = velocity(ball, fps, 45.)
    ball_speed = np.linalg.norm(ball_velocity, axis=-1)
    edge = np.max(np.abs(ball) - np.asarray(tracking.pitch) / 2, axis=1)
    player_velocity = velocity(players, fps, 13.)
    speed = np.linalg.norm(player_velocity, axis=-1)
    distance = np.linalg.norm(players - ball[:, None], axis=-1)
    speed_delta = np.linalg.norm(player_velocity - ball_velocity[:, None], axis=-1)
    features = {
        "ball_seen": observed,
        "since_ball": previous_distance,
        "until_ball": next_distance,
        "ball_speed": ball_speed,
        "ball_speed_context": window_mean(ball_speed, fps // 3),
        "ball_speed_change": np.abs(np.r_[np.nan, np.diff(ball_speed)]) * fps,
        "pitch_edge": edge,
        "last_pitch_edge": np.where(last >= 0, edge[np.maximum(last, 0)], np.nan),
        "next_pitch_edge": np.where(following < len(ball),
                                    edge[np.minimum(following, len(ball) - 1)], np.nan),
    }
    for side, suffix in enumerate(("h", "a")):
        mask = tracking.teams == side
        side_distance = distance[:, mask]
        nearest = np.argmin(np.where(np.isfinite(side_distance), side_distance, np.inf), axis=1)
        closest = side_distance[index, nearest]
        relative = speed_delta[:, mask][index, nearest]
        centroid = mean(players[:, mask], axis=1)
        spread = mean(np.linalg.norm(players[:, mask] - centroid[:, None], axis=-1), axis=1)
        collective = mean(speed[:, mask], axis=1)
        features.update({
            f"near_{suffix}": closest,
            f"relative_{suffix}": relative,
            f"close_count_{suffix}": np.where(observed, (side_distance < 3).sum(axis=1), np.nan),
            f"spread_{suffix}": window_mean(spread, fps),
            f"movement_{suffix}": window_mean(collective, fps),
            f"centroid_speed_{suffix}": window_mean(
                np.linalg.norm(velocity(centroid, fps, 10.), axis=-1), fps),
            f"coverage_{suffix}": np.isfinite(players[:, mask, 0]).sum(axis=1),
        })
    margin = features["near_a"] - features["near_h"]
    features["control_margin"] = margin
    features["control_context"] = window_mean(np.clip(margin, -4., 4.), fps)
    features["last_control"] = np.where(last >= 0, margin[np.maximum(last, 0)], np.nan)
    features["next_control"] = np.where(following < len(ball),
                                        margin[np.minimum(following, len(ball) - 1)], np.nan)
    return features
