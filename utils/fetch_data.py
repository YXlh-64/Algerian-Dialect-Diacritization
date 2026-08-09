#!/usr/bin/env python3
"""
utils/fetch_data.py -- download the dataset from the team's Google Drive
and lay it out at <repo_root>/data so training/track3/*/finetune_*.py can
find it (see README.md: "data should be accessed externally from Google
Drive API only, don't keep track of the dataset in this repository").

Usage:
    python utils/fetch_data.py
    python utils/fetch_data.py --force        # re-download even if ./data already looks valid
    python utils/fetch_data.py --folder-id XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Requires the Drive folder to be shared as "Anyone with the link -> Viewer".
If it's ever locked down to specific accounts instead, this script (which
uses anonymous gdown, no OAuth) will need to be swapped for one using
google-api-python-client + a service account or OAuth flow.
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TMP_DOWNLOAD_DIR = REPO_ROOT / "_gdrive_download_tmp"

# ---------------------------------------------------------------------------
# PLACEHOLDER -- replace with the real Drive folder ID (the id= value from
# the shared folder's URL, e.g. for
#   https://drive.google.com/drive/folders/1AbC2dEfGhIjKlMnOpQrStUvWxYz
# the ID is "1AbC2dEfGhIjKlMnOpQrStUvWxYz").
# ---------------------------------------------------------------------------
DRIVE_FOLDER_ID = "PASTE_DRIVE_FOLDER_ID_HERE"

# Files/folders we expect once the download lands in the right place --
# used both to sanity-check an existing ./data and to locate the real
# dataset root inside whatever nesting gdown produces.
REQUIRED_ENTRIES = ["train_data", "dev_data", "test_data", "vocab.json", "class_labels.txt"]


def _looks_valid(directory: Path) -> bool:
    return directory.is_dir() and all((directory / entry).exists() for entry in REQUIRED_ENTRIES)


def _find_dataset_root(search_root: Path) -> Path | None:
    """The Drive folder is named 'Data' (capital D) and gdown may nest it
    under an extra directory depending on version/options, so search a
    couple of levels deep for the first directory containing all the
    expected entries, rather than assuming a fixed depth or exact name."""
    if _looks_valid(search_root):
        return search_root
    for candidate in search_root.rglob("*"):
        if candidate.is_dir() and _looks_valid(candidate):
            return candidate
    return None


def fetch_data(folder_id: str, force: bool = False) -> Path:
    if _looks_valid(DATA_DIR) and not force:
        print(f"{DATA_DIR} already has {', '.join(REQUIRED_ENTRIES)} -- skipping download. "
              f"Use --force to re-download.")
        return DATA_DIR

    if folder_id == "PASTE_DRIVE_FOLDER_ID_HERE" or not folder_id:
        sys.exit(
            "ERROR: DRIVE_FOLDER_ID is still a placeholder.\n"
            "Open utils/fetch_data.py and replace DRIVE_FOLDER_ID with the real "
            "Google Drive folder ID, or pass it on the command line instead:\n"
            "  - direct:       python utils/fetch_data.py --folder-id XXXX\n"
            "  - via pipeline: python run_pipeline.py ... --drive-folder-id XXXX\n"
            "The folder must be shared as 'Anyone with the link -> Viewer'."
        )

    try:
        import gdown
    except ImportError:
        sys.exit(
            "ERROR: the 'gdown' package is required to fetch data.\n"
            "Install it with: pip install gdown\n"
            "(or just run run_pipeline.py, which installs requirements.txt automatically)"
        )

    if TMP_DOWNLOAD_DIR.exists():
        shutil.rmtree(TMP_DOWNLOAD_DIR)
    TMP_DOWNLOAD_DIR.mkdir(parents=True)

    print(f"Downloading dataset from Google Drive folder {folder_id} ...")
    try:
        gdown.download_folder(
            id=folder_id,
            output=str(TMP_DOWNLOAD_DIR),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        shutil.rmtree(TMP_DOWNLOAD_DIR, ignore_errors=True)
        sys.exit(
            f"ERROR: gdown failed to download the folder ({e}).\n"
            f"Common causes: folder not actually public, wrong folder ID, or "
            f"Google's per-file download quota was hit (retry later, or download "
            f"the folder as a zip manually from the Drive UI and extract it to "
            f"{DATA_DIR} yourself)."
        )

    dataset_root = _find_dataset_root(TMP_DOWNLOAD_DIR)
    if dataset_root is None:
        shutil.rmtree(TMP_DOWNLOAD_DIR, ignore_errors=True)
        sys.exit(
            f"ERROR: downloaded folder doesn't contain the expected structure "
            f"({', '.join(REQUIRED_ENTRIES)}). Check DRIVE_FOLDER_ID points at the "
            f"'Data' folder shown in the team's Drive, not a parent/sibling folder."
        )

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    shutil.move(str(dataset_root), str(DATA_DIR))
    shutil.rmtree(TMP_DOWNLOAD_DIR, ignore_errors=True)

    print(f"Dataset ready at {DATA_DIR}")
    return DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder-id", default=DRIVE_FOLDER_ID,
                         help="Google Drive folder ID to download (overrides the "
                              "DRIVE_FOLDER_ID placeholder in this file).")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if ./data already looks valid.")
    args = parser.parse_args()
    fetch_data(folder_id=args.folder_id, force=args.force)


if __name__ == "__main__":
    sys.exit(main())