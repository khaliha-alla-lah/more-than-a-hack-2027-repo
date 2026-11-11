"""Independent tracking corruptions for external-match experiments."""

import numpy as np
from scipy.signal import lfilter

from .tracking import Tracking


def position_error(rng, shape, sigma, rho):
    noise = rng.normal(0, sigma, shape).astype(np.float32)
    if rho == 0:
        return noise
    # Start in the stationary distribution and preserve marginal variance.
    noise[1:] *= np.sqrt(1 - rho**2)
    return lfilter(np.array([1], dtype=np.float32),
                   np.array([1, -rho], dtype=np.float32), noise, axis=0)


def corrupt(tracking, lengths, seed, *, ball_loss=False, player_loss=False,
            jitter=False, rho=0.9, stress=False):
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    ball_rng, player_rng, noise_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    ]
    ball, players = tracking.ball.copy(), tracking.players.copy()
    if jitter:
        ball += position_error(noise_rng, ball.shape, 0.28 if stress else 0.15, rho)
        players += position_error(noise_rng, players.shape, 0.14 if stress else 0.08, rho)
    if ball_loss:
        lengths = np.asarray(lengths)
        if not lengths.size or np.any(lengths <= 0):
            raise ValueError("Expected positive missing-ball burst lengths")
        mask = np.zeros(len(ball), dtype=bool)
        target = (0.50 if stress else 0.40) * len(mask)
        count = 0
        while count < target:
            length = min(len(mask), int(ball_rng.choice(lengths)))
            start = ball_rng.integers(len(mask) - length + 1)
            span = mask[start:start + length]
            count += length - np.count_nonzero(span)
            span[:] = True
        ball[mask] = np.nan
    if player_loss:
        for slot in range(players.shape[1]):
            for start in player_rng.integers(len(players), size=max(1, len(players) // 1500)):
                players[start:start + player_rng.integers(30, 180), slot] = np.nan
    return Tracking(players, tracking.teams, ball, tracking.pitch, tracking.fps)
