"""One-time script: clone Samsung/CredData and materialize its Python-relevant
source files.

Must run on a native Linux filesystem (WSL is fine) — CredData's own README
states some original repository filenames it reconstructs while building the
dataset are invalid on NTFS. Running this from Windows Python, or against a
/mnt/c/... path even from inside WSL, will fail partway through with
filesystem errors. Once download_data.py finishes, its output (data/ and
meta/) is Windows-safe and gets copied into this project's data/ folder.

Before running download_data.py (which clones every repo in snapshot.json,
all languages), this script rewrites snapshot.json down to only the repos
that have at least one Python-referenced row in meta/*.csv — computed via
CredData's own short-repo-id scheme (CRC32 of the snapshot key, see
get_new_repo_id() in their download_data.py) so we never clone a repo with
zero Python findings in the first place. The same repo set is also used to
move aside non-Python meta/*.csv files, since download_data.py's own
obfuscate_creds() step iterates meta/ independently of snapshot.json and
would otherwise crash looking for files from repos we deliberately skipped.

Usage (from inside WSL):
    wsl
    cd ~
    python3 /mnt/c/Users/krnan/Desktop/Research/credhunter-x/scripts/fetch_creddata.py

Re-run is safe: an existing clone/output is left alone unless removed first.
"""

from __future__ import annotations

import binascii
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CREDDATA_REPO = "https://github.com/Samsung/CredData.git"

# Native Linux path (e.g. WSL's ext4 home) — NOT a /mnt/c/... Windows-mounted path.
WORK_DIR = Path.home() / ".cache" / "credhunter-x" / "CredData"
VENV_DIR = Path.home() / ".cache" / "credhunter-x" / "venv"

# Windows-side project location the final, Windows-safe output is copied to.
PROJECT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "creddata_raw" / "CredData"


def _require_native_linux_filesystem() -> None:
    if platform.system() != "Linux":
        sys.exit(
            "download_data.py must run on a native Linux filesystem (WSL is fine).\n"
            "Run this script from inside WSL, e.g.:\n"
            "  wsl\n"
            "  cd ~ && python3 /mnt/c/Users/krnan/Desktop/Research/credhunter-x"
            "/scripts/fetch_creddata.py"
        )
    if str(WORK_DIR).startswith("/mnt/"):
        sys.exit(
            f"{WORK_DIR} resolves to a Windows-mounted path (still NTFS underneath),\n"
            "even though this is running under WSL. Clone into WSL's native\n"
            "filesystem instead (e.g. under $HOME, not /mnt/c/...)."
        )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_creddata() -> None:
    if WORK_DIR.exists():
        print(f"{WORK_DIR} already exists, skipping clone.")
        return
    WORK_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", CREDDATA_REPO, str(WORK_DIR)])


def compute_python_repo_ids() -> set[str]:
    """Meta filenames (= short repo IDs) that have >=1 row referencing a .py file."""
    ids: set[str] = set()
    for csv_path in (WORK_DIR / "meta").glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["FilePath"].endswith(".py"):
                    ids.add(csv_path.stem)
                    break
    return ids


def filter_snapshot_to_python_repos(python_repo_ids: set[str]) -> None:
    """Rewrite snapshot.json in-place to only the repos we actually need, so
    download_data.py skips cloning everything else. Keeps a backup so this
    is idempotent across re-runs."""
    snapshot_path = WORK_DIR / "snapshot.json"
    backup_path = WORK_DIR / "snapshot_full.json.bak"
    if not backup_path.exists():
        shutil.copy(snapshot_path, backup_path)

    full_snapshot = json.loads(backup_path.read_text())

    def short_id(key: str) -> str:
        return f"{binascii.crc32(binascii.unhexlify(key)):08x}"

    filtered = {k: v for k, v in full_snapshot.items() if short_id(k) in python_repo_ids}
    snapshot_path.write_text(json.dumps(filtered))
    print(
        f"snapshot.json: {len(full_snapshot)} repos -> {len(filtered)} repos "
        "(Python-referenced only)"
    )


def filter_meta_to_python_repos(python_repo_ids: set[str]) -> None:
    """obfuscate_creds() (a step inside download_data.py) iterates every
    meta/*.csv and expects a matching file under data/ — since we only clone
    repos with a Python finding, meta/ must be trimmed to match or it
    crashes on the first excluded repo. Excluded CSVs are moved aside, not
    deleted, so this is idempotent and reversible."""
    meta_dir = WORK_DIR / "meta"
    excluded_dir = WORK_DIR / "meta_excluded_backup"
    excluded_dir.mkdir(exist_ok=True)
    moved = 0
    for csv_path in meta_dir.glob("*.csv"):
        if csv_path.stem not in python_repo_ids:
            shutil.move(str(csv_path), str(excluded_dir / csv_path.name))
            moved += 1
    if moved:
        print(f"Moved {moved} non-Python meta CSVs aside to {excluded_dir}")


def create_venv() -> Path:
    """Isolated venv for CredData's own requirements.txt, kept separate from
    this project's uv-managed venv and from system Python."""
    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    return venv_python


def install_requirements(venv_python: Path) -> None:
    _run([str(venv_python), "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORK_DIR)


def run_download_data(venv_python: Path) -> None:
    # --clean_data: download_data.py refuses to run if data/ already exists
    # (no built-in resume) — this lets it wipe and rebuild data/ cleanly,
    # which is what we want on a re-run after an earlier partial failure.
    _run([str(venv_python), "download_data.py", "--clean_data"], cwd=WORK_DIR)


def copy_output_to_project() -> None:
    for name in ("data", "meta"):
        src = WORK_DIR / name
        dst = PROJECT_RAW_DIR / name
        if not src.is_dir():
            sys.exit(f"Expected {src} to exist after download_data.py — check its output above.")
        if dst.exists():
            shutil.rmtree(dst)
        print(f"Copying {src} -> {dst}")
        shutil.copytree(src, dst)


def main() -> None:
    _require_native_linux_filesystem()
    clone_creddata()
    python_repo_ids = compute_python_repo_ids()
    print(f"{len(python_repo_ids)} repos have at least one Python-referenced meta row")
    filter_snapshot_to_python_repos(python_repo_ids)
    filter_meta_to_python_repos(python_repo_ids)
    venv_python = create_venv()
    install_requirements(venv_python)
    run_download_data(venv_python)
    copy_output_to_project()
    print(f"\nDone. Windows-safe data/ and meta/ now at {PROJECT_RAW_DIR}")
    print(
        f"({WORK_DIR} and {VENV_DIR}, including WORK_DIR's tmp/ dir, can be "
        "deleted once you've confirmed the copy.)"
    )


if __name__ == "__main__":
    main()
