"""Extract a tactical video frame for local debugging."""

import argparse
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from games import GAME, GAMES, require, source_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=int, help="Zero-based tactical frame")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/local-frames"))
    parser.add_argument("--ffmpeg", help="FFmpeg executable, defaults to imageio-ffmpeg")
    args = parser.parse_args()
    if not 0 <= args.frame < source_frames():
        parser.error("frame must be within the tactical recording")
    executable = args.ffmpeg
    if executable is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            parser.error("provide --ffmpeg or install imageio-ffmpeg==0.6.0")

        executable = get_ffmpeg_exe()
    (video,) = require(["video"], data_dir=args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"tactical-{args.frame:06d}.png"
    seconds = args.frame / GAMES[GAME]["fps"]
    with NamedTemporaryFile(dir=args.output_dir, suffix=".png", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-ss", f"{seconds:.6f}", "-i", video,
                        "-map", "0:v:0", "-frames:v", "1", str(pending)], check=True)
        if not pending.stat().st_size:
            raise RuntimeError(f"No frame decoded at {seconds:.3f} seconds")
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
