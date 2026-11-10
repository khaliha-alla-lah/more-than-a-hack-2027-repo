"""Read the public IDSSE tracking XML and retain its labels separately."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .tracking import Tracking

HALVES = ("firstHalf", "secondHalf")
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def paths(directory, match):
    division = "000001" if match in MATCHES[:2] else "000002"
    suffix = f"DFL-COM-{division}_DFL-MAT-{match}.xml"
    return (Path(directory) / f"DFL_02_01_matchinformation_{suffix}",
            Path(directory) / f"DFL_04_03_positions_raw_observed_{suffix}")


def load(directory, match):
    meta, raw = paths(directory, match)
    sides = {"home": 0, "guest": 1}
    teams = {node.get("TeamId"): sides[node.get("Role")] for node in ET.parse(meta).iter("Team")
             if node.get("Role") in sides}
    tracks = {half: [] for half in HALVES}
    balls = {}
    pitch = None
    for _, element in ET.iterparse(raw, events=("end",)):
        if element.tag == "PitchSize":
            pitch = (float(element.attrib["X"]), float(element.attrib["Y"]))
        if element.tag != "FrameSet":
            continue
        half = element.get("GameSection")
        team = element.get("TeamId")
        if half in tracks and (team == "BALL" or team in teams):
            frames = list(element.iter("Frame"))
            ids = np.array([int(frame.attrib["N"]) for frame in frames])
            xy = np.array([(float(frame.attrib["X"]), float(frame.attrib["Y"]))
                           for frame in frames], dtype=np.float32)
            if team == "BALL":
                if half in balls:
                    raise ValueError(f"Duplicate ball FrameSet for {match} {half}")
                labels = np.array([label(frame.attrib) for frame in frames], dtype=np.int8)
                balls[half] = (ids, xy, labels)
            else:
                tracks[half].append((teams[team], ids, xy))
        element.clear()
    if pitch is None or set(balls) != set(HALVES):
        raise ValueError(f"Missing pitch or half in {match}")
    for half in HALVES:
        ids, xy, labels = balls[half]
        if not len(ids) or np.any(np.diff(ids) <= 0):
            raise ValueError(f"Nonmonotonic ball clock in {match} {half}")
        start, stop = ids[0], ids[-1] + 1
        ball = np.full((stop - start, 2), np.nan, dtype=np.float32)
        ball[ids - start] = xy
        truth = np.full(stop - start, 3, dtype=np.int8)
        truth[ids - start] = labels
        players = np.full((stop - start, len(tracks[half]), 2), np.nan, dtype=np.float32)
        mapping = []
        for slot, (side, player_ids, xy) in enumerate(tracks[half]):
            in_half = (player_ids >= start) & (player_ids < stop)
            players[player_ids[in_half] - start, slot] = xy[in_half]
            mapping.append(side)
        if set(mapping) != {0, 1}:
            raise ValueError(f"Missing team in {match} {half}")
        yield Tracking(players, np.array(mapping), ball, pitch), truth


def label(frame):
    status = frame.get("BallStatus")
    if status == "0":
        return 2
    if status == "1":
        return {"1": 0, "2": 1}.get(frame.get("BallPossession"), 3)
    return 3
