"""Download the 14 required public DFL XML files from the pinned revision."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from .external import MATCHES, REVISION, paths
from .release import fingerprint, write_json

BASE = "https://huggingface.co"
REPOSITORY = "pysport/idsse-data"


def verify(path, record):
    """Check size and the hub's LFS SHA-256 or Git blob SHA-1."""
    size = record["size"]
    if Path(path).stat().st_size != size:
        raise ValueError(f"Incomplete DFL input: {path}")
    lfs = record.get("lfs")
    expected = lfs["oid"] if lfs else record["oid"]
    digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"DFL checksum mismatch: {path}; move the invalid file before retrying")


def download(destination, record, opener=urlopen):
    destination = Path(destination)
    if destination.exists():
        verify(destination, record)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/datasets/{REPOSITORY}/resolve/{REVISION}/{destination.name}"
    with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        pending = Path(stream.name)
        try:
            request = Request(url, headers={"User-Agent": "mtah-possession-reproduction/1"})
            with opener(request, timeout=30) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            stream.close()
            verify(pending, record)
            if destination.exists():
                # Another process completed it; verify rather than replace it.
                verify(destination, record)
            else:
                pending.rename(destination)
        finally:
            stream.close()
            pending.unlink(missing_ok=True)


def fetch(directory, opener=urlopen):
    url = f"{BASE}/api/datasets/{REPOSITORY}/tree/{REVISION}"
    with opener(Request(url, headers={"User-Agent": "mtah-possession-reproduction/1"}), timeout=30) as response:
        records = {entry["path"]: entry for entry in json.load(response) if entry["type"] == "file"}
    required = [path for match in MATCHES for path in paths(directory, match)]
    missing = [path.name for path in required if path.name not in records]
    if missing:
        raise ValueError(f"Pinned dataset index is missing required inputs: {missing}")
    for path in required:
        download(path, records[path.name], opener)
        print(f"Verified {path.name}", flush=True)
    receipt = {"repository": REPOSITORY, "revision": REVISION,
               "files": {path.name: fingerprint(path) for path in required}}
    write_json(Path(directory) / "download-manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/idsse"))
    args = parser.parse_args()
    try:
        fetch(args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"DFL download failed: {error}\nExisting verified inputs were preserved.\n")


if __name__ == "__main__":
    main()
