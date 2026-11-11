"""The ordered input contract for the unchanged phase-one baseline."""

FEATURE_NAMES = (
    "ball_seen", "since_ball", "until_ball", "ball_speed", "ball_speed_context",
    "ball_speed_change", "pitch_edge", "last_pitch_edge", "next_pitch_edge",
    "near_h", "relative_h", "close_count_h", "spread_h", "movement_h",
    "centroid_speed_h", "coverage_h",
    "near_a", "relative_a", "close_count_a", "spread_a", "movement_a",
    "centroid_speed_a", "coverage_a",
    "control_margin", "control_context", "last_control", "next_control",
)
COLLECTIVE_NAMES = (
    "movement_h", "movement_a", "centroid_speed_h", "centroid_speed_a",
    "spread_h", "spread_a",
)


def feature_schema(collective):
    return {"all": list(FEATURE_NAMES),
            "status": list(COLLECTIVE_NAMES if collective else FEATURE_NAMES)}
