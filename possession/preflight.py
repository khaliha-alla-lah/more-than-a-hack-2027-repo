"""Check inference or training prerequisites without loading restricted tracking rows."""

import argparse
import importlib
import json
from pathlib import Path

from games import asset_paths

from .release import read_manifest, runtime


def inspect(data_dir=None, model_dir=Path("data/possession-model"),
            dfl_dir=Path("data/idsse"), rules=False, training=False):
    issues = []
    packages = ("numpy", "pandas") if rules else ("numpy", "pandas", "lightgbm")
    for name in packages:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as error:
            issues.append(f"Python package unavailable: {name} ({error})")
    paths = asset_paths(data_dir=data_dir)
    files = {name: {"path": str(paths[name]), "exists": Path(paths[name]).is_file()}
             for name in ("atd_meta", "atd_raw")}
    for entry in files.values():
        if not entry["exists"]:
            issues.append(f"Missing tactical input: {entry['path']}")
    if training:
        from .external import MATCHES, paths as dfl_paths

        external = [path for match in MATCHES for path in dfl_paths(dfl_dir, match)]
        files["dfl"] = {"present": sum(path.is_file() for path in external), "required": len(external)}
        issues.extend(f"Missing DFL input: {path}" for path in external if not path.is_file())
        if Path(model_dir).exists():
            issues.append(f"Release destination already exists: {model_dir}; choose a new directory")
    elif not rules:
        try:
            read_manifest(model_dir)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(str(error))
    return {"ready": not issues, "mode": "training" if training else "rules" if rules else "learned",
            "runtime": runtime(), "inputs": files, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rules", action="store_true")
    mode.add_argument("--training", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dfl", type=Path, default=Path("data/idsse"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/possession-model"))
    args = parser.parse_args()
    result = inspect(args.data, args.model_dir, args.dfl, args.rules, args.training)
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
