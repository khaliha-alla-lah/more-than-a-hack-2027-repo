"""Pinned public DFL input names; usable without numerical dependencies."""

from pathlib import Path

REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def paths(directory, match):
    if match not in MATCHES:
        raise ValueError(f"Unknown DFL match: {match}")
    division = "000001" if match in MATCHES[:2] else "000002"
    suffix = f"DFL-COM-{division}_DFL-MAT-{match}.xml"
    return (Path(directory) / f"DFL_02_01_matchinformation_{suffix}",
            Path(directory) / f"DFL_04_03_positions_raw_observed_{suffix}")
