"""Tracking adapters. Features consume positions in metres on a centred pitch."""

from dataclasses import dataclass

import numpy as np

from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track


@dataclass
class Tracking:
    players: np.ndarray
    teams: np.ndarray
    ball: np.ndarray
    pitch: tuple
    fps: int = 25

    def section(self, start, stop):
        return Tracking(self.players[start:stop], self.teams, self.ball[start:stop],
                        self.pitch, self.fps)


def load_metrica(data_dir=None, source="tactical"):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir, source=source)
    teams = team_of_each_track(meta)
    count = source_frames(source)
    players = np.empty((count, len(teams), 2), dtype=np.float32)
    ball = np.empty((count, 2), dtype=np.float32)
    rows = 0
    with open(raw) as handle:
        for index, line in enumerate(handle):
            frame, tracks, observation = line.split(":")
            if int(frame) != index + 1 or index >= count:
                raise ValueError(f"Unexpected ATD frame {frame} at row {index + 1}")
            values = np.fromstring(tracks.replace(";", ",").rstrip(","),
                                   sep=",", dtype=np.float32)
            if len(values) != len(teams) * 3:
                raise ValueError(f"Unexpected ATD slot count at frame {frame}")
            players[index] = values.reshape(-1, 3)[:, :2]
            ball[index] = observation.split(",")[:2]
            rows += 1
    if rows != count:
        raise ValueError(f"Expected {count} frames, read {rows}")
    pitch = GAMES[GAME]["pitch"]
    scale = np.asarray(pitch, dtype=np.float32)
    return Tracking((players - 0.5) * scale, teams, (ball - 0.5) * scale,
                    pitch, GAMES[GAME]["fps"])
