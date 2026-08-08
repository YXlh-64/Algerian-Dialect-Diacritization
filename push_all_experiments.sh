#!/usr/bin/env bash
#
# push_all_experiments.sh — push ALL experiments in experiments_map.csv as
# ONE branch with ONE commit, instead of one branch per experiment.
#
# USAGE:
#   ./push_all_experiments.sh /path/to/repo /path/to/staging_dir exp/all-experiments

set -euo pipefail

REPO_DIR="${1:?Usage: ./push_all_experiments.sh /path/to/repo /path/to/staging_dir <branch-name>}"
STAGING_DIR="${2:?Usage: ./push_all_experiments.sh /path/to/repo /path/to/staging_dir <branch-name>}"
BRANCH="${3:-exp/all-experiments}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAP_FILE="$SCRIPT_DIR/experiments_map.csv"

STAGING_DIR="$(cd "$STAGING_DIR" && pwd)"

if [ ! -f "$MAP_FILE" ]; then
  echo "Missing $MAP_FILE"
  exit 1
fi

cd "$REPO_DIR"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Branch $BRANCH already exists locally — checking it out."
  git checkout "$BRANCH"
else
  git checkout main
  git pull --ff-only
  git checkout -b "$BRANCH"
fi

copied=()

copy_if_exists () {
  local src="$1" dest="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    copied+=("$dest")
  fi
}

# Mirror the whole staging tree rather than tracking individual per-slug
# files — after consolidation, model.py/evaluate.py/train.py may be shared
# across a whole track/head_type group instead of one file per experiment,
# so copying by folder is simpler and correct either way.
for folder in configs models training evaluation experiments; do
  if [ -d "$STAGING_DIR/$folder" ]; then
    mkdir -p "$folder"
    cp -r "$STAGING_DIR/$folder/." "$folder/"
  fi
done
copy_if_exists "$STAGING_DIR/leaderboard.md" "experiments/leaderboard.md"

git add configs models training evaluation experiments
git commit -m "Add all 12 experiments (track3: linear_head + bilstm_crf_head)"
git push -u origin "$BRANCH"

echo ""
echo "Pushed $BRANCH with all experiments in one commit."
echo "Open a PR for it on GitHub now."
