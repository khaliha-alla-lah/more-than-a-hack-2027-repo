"""Versioned, fingerprinted model releases and local inference receipts."""

import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

from .schema import FEATURE_NAMES, feature_schema

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
PACKAGES = ("numpy", "pandas", "scipy", "lightgbm", "python-dateutil", "tzdata")
# Changes to these files require an explicit new release, even if widths stay equal.
INFERENCE_FILES = (
    "possession/features.py", "possession/schema.py", "possession/tracking.py",
    "possession/learn.py", "possession/decode.py", "possession/__main__.py",
    "games.py", "starter/baseline.py", "starter/load_smart_tagging.py",
    "evaluation/possession_metrics.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def runtime():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}


def source():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = set(INFERENCE_FILES)
    files.update(str(path.relative_to(ROOT)) for path in (ROOT / "possession").glob("*.py"))
    files.update(str(path.relative_to(ROOT)) for path in ROOT.glob("requirements*.txt"))
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "files": {name: sha256(ROOT / name) for name in sorted(files)}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def check_destination(directory):
    if Path(directory).exists():
        raise FileExistsError(f"Release already exists: {directory}. Choose a new --model-dir; "
                              "existing candidates are never overwritten.")


def save_release(directory, models, benchmark, provenance):
    """Write to a staging directory, then publish the complete release in one rename."""
    directory = Path(directory)
    check_destination(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    collective = benchmark["collective_status"]
    schema = feature_schema(collective)
    widths = [len(FEATURE_NAMES)] if len(models) == 1 else [len(schema["status"]), len(FEATURE_NAMES)]
    if len(models) not in (1, 2):
        raise ValueError("Expected one joint model or two binary models")
    with TemporaryDirectory(prefix=".release-", dir=directory.parent) as temporary:
        stage = Path(temporary) / "candidate"
        stage.mkdir()
        entries = []
        for index, (model, width) in enumerate(zip(models, widths)):
            if model.num_feature() != width:
                raise ValueError(f"Model {index} has {model.num_feature()} features; expected {width}")
            name = f"model-{index}.txt"
            model.save_model(str(stage / name))
            entries.append({"file": name, "num_features": width, **fingerprint(stage / name)})
        report = {**benchmark, "models": [entry["file"] for entry in entries]}
        write_json(stage / "benchmark.json", report)
        manifest = {"format_version": FORMAT_VERSION, "labels": ["H", "A", "D"],
                    "feature_schema": schema, "collective_status": collective,
                    "penalties": benchmark["penalties"], "selected": benchmark["selected"],
                    "models": entries, "benchmark": fingerprint(stage / "benchmark.json"),
                    "source": source(), "runtime": runtime(), "training": provenance}
        write_json(stage / "manifest.json", manifest)
        check_destination(directory)
        stage.rename(directory)
    return manifest


def read_manifest(directory, verify_source=True):
    """Validate the contract and file hashes before importing LightGBM or loading tracking."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model release: {path}. Reconstruct it with "
                                "python -m possession.train, or use --rules. "
                                "A legacy benchmark.json alone is not a verified release.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("labels") != ["H", "A", "D"]:
        raise ValueError("Unsupported model release format or label order")
    collective = manifest.get("collective_status")
    if type(collective) is not bool or manifest.get("feature_schema") != feature_schema(collective):
        raise ValueError("Model feature schema does not match this implementation")
    penalties = manifest.get("penalties")
    if penalties is not None and (not isinstance(penalties, list) or len(penalties) != 2 or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in penalties)):
        raise ValueError("Decoder penalties must be null or two finite nonnegative numbers")
    entries = manifest.get("models", [])
    if not isinstance(entries, list) or len(entries) not in (1, 2):
        raise ValueError("Release must contain one joint model or two binary models")
    widths = [len(FEATURE_NAMES)] if len(entries) == 1 else [len(manifest["feature_schema"]["status"]), len(FEATURE_NAMES)]
    names = set()
    for entry, width in zip(entries, widths):
        name = entry.get("file", "")
        if not name or Path(name).name != name or name in names or name in (".", ".."):
            raise ValueError("Model filenames must be distinct files inside the release")
        names.add(name)
        if entry.get("num_features") != width:
            raise ValueError(f"Feature count mismatch for {name}")
        if fingerprint(directory / name) != {key: entry.get(key) for key in ("bytes", "sha256")}:
            raise ValueError(f"Model checksum mismatch: {name}")
    if fingerprint(directory / "benchmark.json") != manifest.get("benchmark"):
        raise ValueError("Benchmark checksum mismatch")
    if verify_source:
        expected = manifest.get("source", {}).get("files", {})
        changed = [name for name in INFERENCE_FILES if expected.get(name) != sha256(ROOT / name)]
        if changed:
            raise ValueError(f"Inference source differs from the release: {', '.join(changed)}. "
                             "Use the recorded source or build a new release.")
    return manifest


def load_models(directory):
    manifest = read_manifest(directory)
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("Learned inference needs LightGBM; install requirements-possession.txt") from error
    models = [lgb.Booster(model_file=str(Path(directory) / entry["file"]))
              for entry in manifest["models"]]
    for model, entry in zip(models, manifest["models"]):
        if model.num_feature() != entry["num_features"]:
            raise ValueError(f"Loaded feature count mismatch: {entry['file']}")
    return models, manifest["penalties"], manifest["collective_status"]


def run_receipt(output, inputs, *, mode, penalties, model_dir=None, rows):
    receipt = {"format_version": FORMAT_VERSION, "mode": mode, "rows": rows,
               "output": {"file": Path(output).name, **fingerprint(output)},
               "inputs": {name: fingerprint(path) for name, path in inputs.items()},
               "penalties": penalties, "source": source(), "runtime": runtime()}
    if model_dir is not None:
        receipt["model_manifest"] = fingerprint(Path(model_dir) / "manifest.json")
    return receipt
