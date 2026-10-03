"""Pinned, checksummed external data.

Every dataset used in the paper is taken from a published Python wheel on PyPI,
pinned by version and SHA-256, so `python reproduce.py` can fetch it on any
machine with PyPI access. Nothing is redistributed in this repository.

| Data | Wheel | Original source |
|---|---|---|
| NL day-ahead prices 2015-2024 (hourly) | ev2gym==2.0.0 | ENTSO-E Transparency Platform, as bundled by EV2Gym |
| Residential load profiles (25 homes, 15 min, 1 year) | ev2gym==2.0.0 | Pecan Street Dataport, as bundled by EV2Gym |
| ACN-Data charging sessions (Caltech, JPL; 2019-2021) | sustaingym==0.1.7 | ACN-Data (Lee et al., e-Energy 2019), as bundled by SustainGym (CC BY 4.0) |
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache"

WHEELS = {
    "ev2gym-2.0.0-py3-none-any.whl":
        "b1f22280d9ae9da93bb26031730412d952845014cd2718f92bc030b78cddcb58",
    "sustaingym-0.1.7-py3-none-any.whl":
        "a791cea8b3bf226056e185cf26be6aae552799325b4a8a7c876e0509ce53f0a0",
}
FILES = {
    "ev2gym/data/Netherlands_day-ahead-2015-2024.csv":
        "ed29bebc6c47c177939ffb5468f253be6d664cc819412a0e446dc6aec7cc1828",
    "ev2gym/data/residential_loads.csv":
        "276246b5c1f47e02e3a22b535851d754f28c4902e3ae28e28cbee55cfd1553ea",
    "sustaingym/data/evcharging/acn_data/caltech/2019-05-01 2019-08-31.csv.gz":
        "d23a07e3b5175dca0da107610557f604fa177f71bfc696687c70a0470964047b",
    "sustaingym/data/evcharging/acn_data/caltech/2019-09-01 2019-12-31.csv.gz":
        "c9181c918e2a834df4639e6c3e29328ed776d61419c9c89108e3c9e2b649919d",
    "sustaingym/data/evcharging/acn_data/caltech/2020-02-01 2020-05-31.csv.gz":
        "794ad5e6bb8cf10d31ecaa7296a66aa171ee4ed1c9ab81933c6b4eeb3c38649e",
    "sustaingym/data/evcharging/acn_data/caltech/2021-05-01 2021-08-31.csv.gz":
        "3d35712b2e390104425639bf00113189aac3ad1851c71c9a4a6f35e792f92414",
    "sustaingym/data/evcharging/acn_data/jpl/2019-05-01 2019-08-31.csv.gz":
        "b8e602313ce5bcfd37cd6565075754544bf0987e570bd543037749bc6d62f057",
    "sustaingym/data/evcharging/acn_data/jpl/2019-09-01 2019-12-31.csv.gz":
        "f420bb0066e75c7febaa2bfd27112fdc65e60765049bde3615f727ff9ae6ed55",
    "sustaingym/data/evcharging/acn_data/jpl/2020-02-01 2020-05-31.csv.gz":
        "4dcad95f2983f75b47c3ea67fda9992dec03f2d4c5d332b44083880c66d8116c",
    "sustaingym/data/evcharging/acn_data/jpl/2021-05-01 2021-08-31.csv.gz":
        "5b0802a20146f0a4b553d2e7eae2c5b7f3a91f195bcb1caaabe02778f3ec70e5",
}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(verbose: bool = True) -> pathlib.Path:
    """Download the pinned wheels (if needed), extract the data files and verify
    every checksum. Returns the extraction root."""
    wheel_dir = CACHE / "wheels"
    out = CACHE / "extracted"
    if all((out / f).exists() and _sha256(out / f) == h for f, h in FILES.items()):
        return out
    wheel_dir.mkdir(parents=True, exist_ok=True)
    missing = [w for w in WHEELS if not (wheel_dir / w).exists()]
    if missing:
        specs = [w.split("-py3")[0].replace("-", "==", 1) for w in missing]
        if verbose:
            print("downloading", ", ".join(specs), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q",
                        "-d", str(wheel_dir), *specs], check=True)
    for w, h in WHEELS.items():
        got = _sha256(wheel_dir / w)
        if got != h:
            raise RuntimeError(f"checksum mismatch for {w}: {got}")
    for w in WHEELS:
        with zipfile.ZipFile(wheel_dir / w) as z:
            for f in FILES:
                if f in z.namelist():
                    z.extract(f, out)
    for f, h in FILES.items():
        got = _sha256(out / f)
        if got != h:
            raise RuntimeError(f"checksum mismatch for {f}: {got}")
    if verbose:
        print(f"data verified ({len(FILES)} files) in {out}", flush=True)
    return out
