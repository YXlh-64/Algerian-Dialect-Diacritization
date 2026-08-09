#!/usr/bin/env python3
"""
run_pipeline.py -- single entrypoint to run any track/head_type/model
combination in this repo, e.g.:

    python run_pipeline.py --track track3 --head-type bilstm_crf_head --model camelbert_mix
    python run_pipeline.py --track track3 --head-type linear_head --model marbertv2 --data-dir ./data

Design notes (read this before "fixing" it):

- This is a thin DISPATCHER, not a namespace-injection hack. It finds the
  right training/<track>/<head_type>/finetune_<head_type>.py by naming
  convention alone and runs it as a subprocess -- it does not import,
  patch, or inject anything into that script's globals. The actual
  cross-file dependency wiring (models/*.py, evaluation/*.py) lives inside
  each finetune_*.py itself as real `import` statements. That's a
  deliberate choice: explicit imports in the training scripts are
  debuggable, IDE/linter-friendly, and work even if someone runs a
  finetune_*.py directly without going through this dispatcher at all.
  This file's only job is convenience + consistent CLI surface across
  tracks, not correctness -- correctness is each script's own.

- "head_type" here (linear_head / bilstm_crf_head) is the architecture-head
  axis, NOT the same as "Strategy A/B/C/D" (pretraining regime) described
  in experiments/README.md and the team's HOW_TO_ORGANIZE_EXPERIMENTS.md.
  Don't conflate them -- see that doc's "three separate axes" section.

- New tracks/head-types need zero changes here: just add
  training/<track>/<head_type>/finetune_<head_type>.py following the same
  convention and it's auto-discovered.

- "--model" is forwarded as --active-model to the underlying script's own
  argparse. Once configs/*.yaml are wired to a real loader (still a
  TODO -- see configs/README.md), a --config flag can be added the same
  way, forwarded straight through.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = REPO_ROOT / "training"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
# Marker records the hash of requirements.txt at install time, so editing
# requirements.txt later automatically triggers a re-install on next run --
# this is NOT just a one-time "ran before" flag.
INSTALL_MARKER = REPO_ROOT / ".deps_installed"
# Records which torch build ("cuda" or "cpu") is currently installed, so a
# later run on the same machine can tell "already correct, skip" apart from
# "GPU availability changed since last run, reinstall the right build".
TORCH_VARIANT_MARKER = REPO_ROOT / ".torch_variant"
# NOTE: PyTorch's CUDA wheel indexes get retired for newer torch/Python
# versions over time (cu121 had no wheels left for Python 3.13 as of
# mid-2026, silently falling back to a CPU-only build even on machines
# with a working GPU). If this default 404s for you, run `nvidia-smi`
# and look at the "CUDA Version" in the top-right of its output -- that's
# the *maximum* CUDA version your driver supports, so any PyTorch index
# at or below it will work. Check https://pytorch.org/get-started/locally/
# for the current list of index names (cu126, cu128, cu129, ...), then
# override with --cuda-index instead of editing this file.
DEFAULT_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def _requirements_hash():
    return hashlib.sha256(REQUIREMENTS_FILE.read_bytes()).hexdigest()


def _gpu_available() -> bool:
    """Detect an NVIDIA GPU via nvidia-smi, without needing torch already
    installed (nvidia-smi ships with the driver, not with torch/CUDA)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _installed_torch_variant():
    """Returns 'cuda', 'cpu', or None (not installed), based on the
    currently-importable torch, not just the marker file -- so a torch
    installed/removed by hand outside this script is still detected."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return None


def ensure_torch(skip: bool, force: bool, cpu_only: bool, cuda_index: str):
    """Install the torch build that matches the machine: CUDA build if an
    NVIDIA GPU is detected (via nvidia-smi), CPU build otherwise.

    Skipped entirely with --skip-torch. Re-checks cheaply on every run (a
    single nvidia-smi call) and only re-installs when the desired variant
    actually differs from both the marker AND the currently-importable
    torch -- so normal runs pay no extra cost once it's correct. Any
    existing torch is uninstalled first so switching machines/variants
    never leaves a mismatched CPU+CUDA torch install fighting each other.
    """
    if skip:
        return

    want = "cpu" if cpu_only else ("cuda" if _gpu_available() else "cpu")
    have_marker = TORCH_VARIANT_MARKER.read_text().strip() if TORCH_VARIANT_MARKER.exists() else None
    have_actual = _installed_torch_variant()

    if not force and have_marker == want and have_actual == want:
        return  # correct build already installed, nothing to do

    print(f"{'GPU detected' if want == 'cuda' else 'No GPU detected (or --cpu-only)'} "
          f"-> installing {want.upper()} build of torch ...")

    if have_actual is not None:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if want == "cuda":
        cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", cuda_index]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: CUDA torch install from {cuda_index} failed -- falling back "
                  f"to the default (non-index) pip install. If your GPU/driver needs a "
                  f"different CUDA version, pass --cuda-index (see "
                  f"https://pytorch.org/get-started/locally/ for the right index URL for "
                  f"your setup).", file=sys.stderr)
            result = subprocess.run([sys.executable, "-m", "pip", "install", "torch"])
    else:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "torch"])

    if result.returncode != 0:
        print("ERROR: torch installation failed (see pip output above). "
              "Fix the issue and re-run, or pass --skip-torch to bypass this step "
              "(and install torch yourself).", file=sys.stderr)
        sys.exit(result.returncode)

    # Record what's ACTUALLY installed, not what we asked pip for -- the
    # fallback "pip install torch" above pulls a CUDA-capable build from
    # PyPI on Linux even when the --index-url install failed, so trusting
    # `want` here would mislabel a working CUDA install as "cpu" and cause
    # every future run to pointlessly uninstall/reinstall it.
    #
    # This has to run in a FRESH subprocess, not call _installed_torch_variant()
    # in-process: if torch was already imported once in this process (e.g. by
    # the pre-install check above), Python caches it in sys.modules, and a
    # second in-process `import torch` after pip swaps the files on disk
    # would just return that stale cached module instead of the new install.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
        capture_output=True, text=True,
    )
    actual = probe.stdout.strip() if probe.returncode == 0 and probe.stdout.strip() in ("cuda", "cpu") else want
    TORCH_VARIANT_MARKER.write_text(actual)
    print(f"torch ({actual}) installed.\n")


def ensure_requirements(skip_install: bool, force_reinstall: bool):
    """Install requirements.txt via pip on first run only.

    Skipped entirely with --skip-install. Re-runs automatically if
    requirements.txt changes (hash mismatch) or if --reinstall is passed.
    """
    if skip_install:
        return
    if not REQUIREMENTS_FILE.exists():
        sys.exit(
            f"ERROR: {REQUIREMENTS_FILE.name} not found at {REQUIREMENTS_FILE}.\n"
            f"This repo's dependencies (transformers, gdown, etc.) can't be installed "
            f"without it, so continuing would just fail later with a confusing "
            f"ImportError mid-run. Either restore {REQUIREMENTS_FILE.name}, or pass "
            f"--skip-install if you've already set up the environment yourself."
        )

    current_hash = _requirements_hash()
    if not force_reinstall and INSTALL_MARKER.exists():
        if INSTALL_MARKER.read_text().strip() == current_hash:
            return  # already installed for this exact requirements.txt

    print(f"Installing dependencies from {REQUIREMENTS_FILE.name} (first run or "
          f"requirements changed) ...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: dependency installation failed (see pip output above). "
              "Fix the issue and re-run, or pass --skip-install to bypass this step.",
              file=sys.stderr)
        sys.exit(result.returncode)

    INSTALL_MARKER.write_text(current_hash)
    print("Dependencies installed.\n")


def ensure_data(skip_fetch: bool, force_fetch: bool, folder_id: str = None):
    """Make sure <repo_root>/data exists and looks valid, fetching it from
    Google Drive via utils/fetch_data.py if it doesn't.

    Skipped entirely with --skip-data-fetch (e.g. you're using --data-dir
    to point at a dataset that already lives somewhere else). Re-downloads
    if --force-data-fetch is passed even when ./data already looks valid.
    """
    if skip_fetch:
        return
    try:
        from utils.fetch_data import fetch_data, DRIVE_FOLDER_ID, _looks_valid, DATA_DIR
    except ImportError as e:
        print(f"NOTE: couldn't import utils/fetch_data.py ({e}), skipping data fetch.",
              file=sys.stderr)
        return

    if not force_fetch and _looks_valid(DATA_DIR):
        return  # already present, nothing to do

    fetch_data(folder_id=folder_id or DRIVE_FOLDER_ID, force=force_fetch)


def discover_runs():
    """Scan training/<track>/<head_type>/finetune_<head_type>.py -> {(track, head_type): path}.
    Pure filesystem convention, no hardcoded track/head_type list."""
    runs = {}
    if not TRAINING_DIR.exists():
        return runs
    for track_dir in sorted(TRAINING_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        for head_dir in sorted(track_dir.iterdir()):
            if not head_dir.is_dir():
                continue
            expected = head_dir / f"finetune_{head_dir.name}.py"
            if expected.exists():
                runs[(track_dir.name, head_dir.name)] = expected
    return runs


def main():
    runs = discover_runs()

    parser = argparse.ArgumentParser(
        description="Run a track/head_type/model combination.",
        epilog=(
            "Available (track, head_type) combinations:\n  "
            + "\n  ".join(f"{t}/{h}" for t, h in sorted(runs)) if runs
            else "No finetune_<head_type>.py scripts found under training/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--track", required=True, help="e.g. track3")
    parser.add_argument("--head-type", required=True,
                         help="e.g. linear_head, bilstm_crf_head (the architecture-head axis -- "
                              "NOT Strategy A/B/C/D, see experiments/README.md)")
    parser.add_argument("--model", required=True,
                         help="MODEL_REGISTRY key, forwarded as --active-model "
                              "(e.g. camelbert_mix, arabert_v02, marbertv2)")
    parser.add_argument("--data-dir", default=None,
                         help="Local dataset root (containing train_data/, dev_data/, test_data/). "
                              "If omitted, the script falls back to /kaggle/input auto-detection, "
                              "then ./data relative to wherever the underlying script runs from.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved command instead of running it.")
    parser.add_argument("--skip-install", action="store_true",
                         help="Don't check/install requirements.txt before running.")
    parser.add_argument("--reinstall", action="store_true",
                         help="Force re-running pip install -r requirements.txt "
                              "even if the install marker is up to date.")
    parser.add_argument("--skip-data-fetch", action="store_true",
                         help="Don't check/fetch ./data via utils/fetch_data.py before running "
                              "(e.g. when using --data-dir to point elsewhere).")
    parser.add_argument("--force-data-fetch", action="store_true",
                         help="Re-download the dataset from Drive even if ./data already looks valid.")
    parser.add_argument("--drive-folder-id", default=None,
                         help="Override the DRIVE_FOLDER_ID placeholder in utils/fetch_data.py "
                              "without editing the file.")
    parser.add_argument("--skip-torch", action="store_true",
                         help="Don't auto-detect/install torch before running (use whatever's "
                              "already installed).")
    parser.add_argument("--force-torch", action="store_true",
                         help="Re-run the torch install even if the correct build already "
                              "looks installed.")
    parser.add_argument("--cpu-only", action="store_true",
                         help="Force the CPU build of torch even if a GPU is detected.")
    parser.add_argument("--cuda-index", default=DEFAULT_CUDA_INDEX,
                         help=f"pip --index-url to use for the CUDA torch build "
                              f"(default: {DEFAULT_CUDA_INDEX}; pick the right one for your "
                              f"driver at https://pytorch.org/get-started/locally/ if this "
                              f"one fails).")
    args, passthrough = parser.parse_known_args()

    # Resolve which script this invocation maps to (and validate it exists)
    # BEFORE running any of the ensure_* side effects below -- this is what
    # makes --dry-run actually dry: on a fresh machine, ensure_torch/
    # ensure_requirements/ensure_data would otherwise trigger a real
    # multi-GB torch install, a pip install, and a Google Drive fetch just
    # to "preview" a command.
    key = (args.track, args.head_type)
    if key not in runs:
        available = ", ".join(f"{t}/{h}" for t, h in sorted(runs)) or "(none found)"
        parser.error(
            f"No training script for track={args.track!r} head_type={args.head_type!r}.\n"
            f"Available: {available}\n"
            f"(Expected training/{args.track}/{args.head_type}/finetune_{args.head_type}.py)"
        )

    script_path = runs[key]

    cmd = [sys.executable, str(script_path), "--active-model", args.model, *passthrough]

    # The underlying scripts auto-detect /kaggle/input first, then fall back
    # to a `./data` PATH RELATIVE TO THEIR OWN CWD -- not a real --data-dir
    # flag (that's a real gap; see configs/README.md TODO). Until that's
    # wired as a proper CLI flag inside the finetune_*.py scripts themselves,
    # the honest way to point at a local dataset is to run the subprocess
    # with cwd set to the parent of a `data/` folder.
    run_cwd = REPO_ROOT
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.exists():
            parser.error(f"--data-dir does not exist: {data_dir}")
        if data_dir.name != "data":
            print(f"NOTE: underlying scripts look for ./data specifically, but "
                  f"--data-dir points at {data_dir} (dir name {data_dir.name!r} != 'data'). "
                  f"Symlinking or renaming may be needed if it isn't auto-detected.",
                  file=sys.stderr)
        run_cwd = data_dir.parent if data_dir.name == "data" else data_dir

    print(f"Resolved run: track={args.track} head_type={args.head_type} model={args.model}")
    print(f"Script       : {script_path.relative_to(REPO_ROOT)}")
    print(f"Working dir  : {run_cwd}")
    print(f"Command      : {' '.join(cmd)}")

    if args.dry_run:
        return 0

    ensure_torch(skip=args.skip_torch, force=args.force_torch,
                 cpu_only=args.cpu_only, cuda_index=args.cuda_index)
    ensure_requirements(skip_install=args.skip_install, force_reinstall=args.reinstall)

    # If --data-dir was given, the caller is pointing at their own dataset
    # location -- don't also try to fetch/overwrite ./data.
    ensure_data(
        skip_fetch=args.skip_data_fetch or bool(args.data_dir),
        force_fetch=args.force_data_fetch,
        folder_id=args.drive_folder_id,
    )

    result = subprocess.run(cmd, cwd=run_cwd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())