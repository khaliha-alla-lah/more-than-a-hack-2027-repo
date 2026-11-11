"""Reconstruct and freeze the historical DFL baseline, without generating a submission."""

import argparse
from pathlib import Path

from games import require

from .release import check_destination


def train(data_dir, dfl_dir, model_dir):
    check_destination(model_dir)
    from .external import MATCHES, paths as dfl_paths

    missing = [str(path) for match in MATCHES for path in dfl_paths(dfl_dir, match)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DFL inputs. Run python -m possession.fetch_dfl --output "
                                f"{dfl_dir}\n" + "\n".join(missing))
    try:
        import lightgbm  # noqa: F401 -- check the runtime before loading tracking
    except ImportError as error:
        raise RuntimeError("Training needs LightGBM; install requirements-possession.txt") from error
    from .benchmark import fit_and_evaluate
    from .tracking import load_metrica

    paths = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    result = fit_and_evaluate(dfl_dir, load_metrica(data_dir), model_dir,
                              dict(zip(("metadata", "tracking"), paths)))
    print(f"Frozen release: {model_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/possession-model"))
    args = parser.parse_args()
    try:
        train(args.data, args.dfl, args.model_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training error: {error}\n")


if __name__ == "__main__":
    main()
