"""Reconstruct exact model bytes from repository chunks; verify all SHA-256 hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    records = json.loads((ROOT / "weights/manifest.json").read_text())
    for record in records:
        target = ROOT / record["target"]
        if target.exists() and sha(target) == record["sha256"]:
            print("Verified", record["target"], flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as out:
            for part in record["parts"]:
                source = ROOT / part["path"]
                assert source.stat().st_size == part["size"] and sha(source) == part["sha256"], source
                with source.open("rb") as f:
                    for block in iter(lambda:f.read(4*1024**2), b""):
                        out.write(block)
        assert tmp.stat().st_size == record["size"] and sha(tmp) == record["sha256"], target
        tmp.replace(target)
        print("Restored", record["target"], flush=True)


if __name__ == "__main__":
    main()
