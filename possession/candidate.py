"""Feature preparation and prediction for the validated motion candidate."""
import numpy as np
from .vendor.sam.ball_tracking import reconstruct_ball_trajectory, compute_ball_player_kinematics
from .vendor.sam.gbm_features import extract_dead_features, extract_team_features, extract_transition_features
from .vendor.sam.gbm_model import PossessionGBMBundle as GBM, CalibrationParams, decode_match_sequence
from .vendor.sam.team_dynamics import compute_team_dynamics


def robust_velocity(points, fps, limit):
    """Local linear fit over seven consecutive observations, preserving gaps."""
    result = np.full_like(points, np.nan)
    if len(points) < 7:
        return result
    slope = np.zeros_like(points[3:-3])
    valid = np.ones(points[3:-3].shape[:-1], dtype=bool)
    for offset in range(-3, 4):
        window = points[3 + offset:len(points) - 3 + offset]
        valid &= np.isfinite(window).all(axis=-1)
        slope += offset * np.nan_to_num(window)
    slope *= fps / 28.
    valid &= np.linalg.norm(slope, axis=-1) <= limit
    slope[~valid] = np.nan
    result[3:-3] = slope
    return result


def prepare(t):
    pitch, fps = t.pitch, t.fps
    blocks = [(1, 0, len(t.ball) - 1)]
    shift = np.asarray(pitch, dtype=np.float32) / 2
    players = t.players + shift
    vel = robust_velocity(players, fps, 13.)
    ball = reconstruct_ball_trajectory(t.ball + shift, None, pitch=pitch, fps=fps)
    kin = compute_ball_player_kinematics(players, vel, t.teams, ball, fps=fps)
    dynamics = compute_team_dynamics(players, np.linalg.norm(vel, axis=-1), vel, t.teams,
                                     ball["kin_xy"], blocks, pitch, fps=fps)
    dead = extract_dead_features(ball, dynamics, blocks, fps=fps)
    return ball, kin, dynamics, blocks, pitch, fps, dead


def team_features(row, pd):
    ball, kin, dynamics, blocks, pitch, fps, _ = row
    return extract_team_features(ball, kin, dynamics, pd, blocks, pitch, fps=fps)


def turn_features(row, pd, ph):
    ball, kin, _, blocks, _, fps, _ = row
    return extract_transition_features(ball, kin, pd, ph, blocks, fps=fps)


def team_predict(model, features):
    return GBM.predict_team_with_booster(
        model, features["X_team"], features["team_swap_pairs"], features["team_negate_cols"])


def predict(tracking, models):
    row = prepare(tracking)
    dead = GBM.predict_dead_with_booster(models[0], row[-1]["X_dead"])
    home = team_predict(models[1], team_features(row, dead))
    turnover = GBM.predict_turn_with_booster(models[2], turn_features(row, dead, home)["X_turn"])
    labels = decode_match_sequence(dead, home, turnover, row[0], row[3],
                                   CalibrationParams(), fps=row[5])
    probabilities = np.column_stack(((1-dead)*home, (1-dead)*(1-home), dead))
    return labels, probabilities, turnover
