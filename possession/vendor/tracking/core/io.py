# See possession/vendor/tracking/PROVENANCE.md for source and adaptation notes.
"""Read the challenge's FIFA/EPTS tracking into per-frame coordinates in metres.

The metadata's DataFormatSpecification defines channel order for each frame range.
Missing player and ball observations remain NaN.
"""

import io as _io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1


@dataclass
class Match:
    """One recording. Frame i of every array is video frame i (submission row i)."""

    x: np.ndarray            # [T, P] metres along the pitch length, NaN where not tracked
    y: np.ndarray            # [T, P] metres along the width
    speed: np.ndarray        # [T, P] m/s, NaN where the source has no speed channel
    team: np.ndarray         # [P] HOME or AWAY
    ball_x: np.ndarray       # [T] metres, NaN where the ball is not seen
    ball_y: np.ndarray       # [T]
    pitch: tuple             # (length, width) in metres
    fps: int = 25
    halves: list = field(default_factory=list)   # [(start, stop)] half-open graded frame spans
    name: str = ""

    @property
    def n_frames(self):
        return len(self.ball_x)


# -- EPTS --------------------------------------------------------------------------------------

def _strip(tag):
    return tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [el for el in root.iter() if _strip(el.tag) == name]


def read_epts_meta(meta_path):
    """Teams, players, channel map and format blocks from a FIFA/EPTS metadata XML."""
    root = ET.parse(meta_path).getroot()

    teams = [t.get("id") for t in _find_all(root, "Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    team_names = []
    for t in _find_all(root, "Team"):
        name = next((c.text for c in t if _strip(c.tag) == "Name"), None)
        team_names.append(name or t.get("id"))

    players = []      # player ids in metadata order
    player_team = {}
    for p in _find_all(root, "Player"):
        pid = p.get("id")
        if pid is None or p.get("teamId") is None:
            continue
        players.append(pid)
        # First team listed is home -- the FIFA/EPTS convention, and the one the challenge's
        # own baseline relies on.
        player_team[pid] = HOME if p.get("teamId") == teams[0] else AWAY

    channel_player = {}   # playerChannelId -> (player id, channel name)
    for ch in _find_all(root, "PlayerChannel"):
        channel_player[ch.get("id")] = (ch.get("playerId"), ch.get("channelId"))

    pitch = None
    for fs in _find_all(root, "FieldSize"):
        vals = {_strip(c.tag): c.text for c in fs}
        try:
            pitch = (float(vals["Width"]), float(vals["Height"]))
        except (KeyError, TypeError, ValueError):
            pass

    fps = 25
    for fr in _find_all(root, "FrameRate"):
        try:
            fps = int(float(fr.text))
        except (TypeError, ValueError):
            pass

    blocks = []
    for spec in _find_all(root, "DataFormatSpecification"):
        start, end = int(spec.get("startFrame")), int(spec.get("endFrame"))
        slots, ball = [], []
        # Children of the spec, in order: frameCount, the ';'-split of players, the ball group.
        for reg in spec:
            kind = _strip(reg.tag)
            if kind != "SplitRegister":
                continue
            inner = [c for c in reg if _strip(c.tag) == "SplitRegister"]
            if inner:   # the player list
                for group in inner:
                    refs = [c.get("playerChannelId") for c in group
                            if _strip(c.tag) == "PlayerChannelRef"]
                    pid = channel_player[refs[0]][0] if refs and refs[0] in channel_player \
                        else None
                    chans = [channel_player.get(r, (None, r))[1] for r in refs]
                    slots.append((pid, chans))
            else:
                ball = [c.get("channelId") for c in reg if _strip(c.tag) == "BallChannelRef"]
        blocks.append({"start": start, "end": end, "slots": slots, "ball": ball})

    return {"teams": teams, "team_names": team_names, "players": players,
            "player_team": player_team, "pitch": pitch, "fps": fps, "blocks": blocks}


def _parse_lines(lines, n_cols):
    """Numeric array [n, n_cols] from raw EPTS lines, all separators flattened to commas.

    Every line must carry exactly n_cols fields. pandas would pad a short line with trailing
    NaNs, so a group written as a bare "NaN" would silently shift every later field one column
    left -- that is checked here, line by line, and raised.
    """
    counts = np.array([ln.count(",") + ln.count(";") + ln.count(":") + 1 for ln in lines])
    bad = np.flatnonzero(counts != n_cols)
    if len(bad):
        i = int(bad[0])
        raise ValueError("line {!r}... has {} fields, the format spec says {} ({} such lines)"
                         .format(lines[i][:40], counts[i], n_cols, len(bad)))
    text = "".join(lines).replace(":", ",").replace(";", ",")
    frame = pd.read_csv(_io.StringIO(text), header=None, na_values=["NaN", "nan", ""],
                        dtype=np.float64, engine="c")
    if frame.shape[1] != n_cols:
        raise ValueError("expected {} fields per line, found {}".format(n_cols, frame.shape[1]))
    return frame.to_numpy()


def _expand_bare_nan(lines, slot_sizes, ball_size):
    """Lines whose missing groups are written as one bare 'NaN' (or left empty) rewritten with one
    NaN per channel, as the format spec lays them out. Lines that already match are untouched,
    and anything else is left for _parse_lines to reject -- this never guesses a column."""
    want = 1 + sum(slot_sizes) + ball_size

    def group(text, size):
        return ",".join(["NaN"] * size) if text.strip().lower() in ("nan", "") else text

    out = []
    for ln in lines:
        if ln.count(",") + ln.count(";") + ln.count(":") + 1 != want:
            parts = ln.rstrip("\n").split(":")
            if len(parts) == 3:
                slots = parts[1].split(";")
                if len(slots) == len(slot_sizes):
                    ln = "{}:{}:{}\n".format(
                        parts[0], ";".join(group(g, n) for g, n in zip(slots, slot_sizes)),
                        group(parts[2], ball_size) if ball_size else parts[2])
        out.append(ln)
    return out


def _channel_target(chan, x, y, speed):
    """Which array a channel id fills. Tolerant of naming: 'x', 'player3_x', 'speed', 's'."""
    name = (chan or "").lower().rsplit("_", 1)[-1]
    if name == "x":
        return x
    if name == "y":
        return y
    if name in ("s", "speed", "v", "velocity"):
        return speed
    return None


def load_epts(meta_path, raw_path, pitch=None, n_frames=None, name=""):
    """A Match from a FIFA/EPTS pair. frame_id 1 becomes video frame 0.

    `pitch` overrides the metadata's field size (the challenge measured 105.3 x 68.0 m against
    the metadata's 105 x 68). `n_frames` fixes the output length; frames with no record stay NaN.
    """
    meta = read_epts_meta(meta_path)
    pitch = pitch or meta["pitch"] or (105.0, 68.0)
    players = meta["players"]
    col_of = {pid: i for i, pid in enumerate(players)}
    team = np.array([meta["player_team"][pid] for pid in players], dtype=np.int8)

    with open(raw_path) as handle:
        lines = handle.readlines()
    lines = [ln.rstrip() + "\n" for ln in lines if ln.strip()]     # CRLF, trailing spaces
    ids = np.array([int(ln[:ln.index(":")]) for ln in lines])
    if n_frames is None:
        n_frames = int(ids.max())

    P = len(players)
    x = np.full((n_frames, P), np.nan, dtype=np.float32)
    y = np.full((n_frames, P), np.nan, dtype=np.float32)
    s = np.full((n_frames, P), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    covered = np.zeros(len(ids), dtype=bool)
    for block in meta["blocks"]:
        covered |= (ids >= block["start"]) & (ids <= block["end"])
    if not covered.all():
        raise ValueError("{} tracking records (first frame_id {}) fall outside every "
                         "DataFormatSpecification block of {}".format(
                             int((~covered).sum()), int(ids[~covered][0]), meta_path))
    outside = (ids < 1) | (ids > n_frames)
    if outside.any():
        raise ValueError("{} tracking records have frame_id outside 1..{} (first {}); is this "
                         "the right recording?".format(int(outside.sum()), n_frames,
                                                       int(ids[outside][0])))
    unknown = [pid for b in meta["blocks"] for pid, _ in b["slots"] if pid not in col_of]
    if unknown:
        raise ValueError("format spec references players missing from the metadata: {}"
                         .format(sorted(set(map(str, unknown)))[:5]))

    for block in meta["blocks"]:
        sel = np.flatnonzero((ids >= block["start"]) & (ids <= block["end"]))
        if not len(sel):
            continue
        sizes = [len(ch) for _, ch in block["slots"]]
        n_cols = 1 + sum(sizes) + len(block["ball"])
        data = _parse_lines(_expand_bare_nan([lines[i] for i in sel], sizes, len(block["ball"])),
                            n_cols)
        rows = data[:, 0].astype(np.int64) - 1
        col = 1
        for pid, chans in block["slots"]:
            j = col_of.get(pid)
            for c, chan in enumerate(chans):
                if j is not None:
                    target = _channel_target(chan, x, y, s)
                    if target is not None:
                        target[rows, j] = data[:, col + c]
            col += len(chans)
        for c, chan in enumerate(block["ball"]):
            target = _channel_target(chan, bx, by, None)
            if target is not None:
                target[rows] = data[:, col + c]

    length, width = pitch
    return Match(x=x * length, y=y * width, speed=s, team=team,
                 ball_x=bx * length, ball_y=by * width, pitch=tuple(pitch), fps=meta["fps"],
                 name=name or os.path.basename(raw_path))
