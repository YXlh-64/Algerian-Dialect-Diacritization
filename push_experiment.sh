#!/usr/bin/env bash
#
# push_experiment.sh — Phase 2: push ONE already-organized experiment's
# files as its own branch.
#
# USAGE:
#   ./push_experiment.sh /path/to/repo /path/to/staging_dir track3 linear_head arabert_v02_09157

set -euo pipefail

REPO_DIR="${1:?Usage: ./push_experiment.sh /path/to/repo /path/to/staging_dir <track> <head_type> <slug>}"
STAGING_DIR="${2:?Usage: ./push_experiment.sh /path/to/repo /path/to/staging_dir <track> <head_type> <slug>}"
TRACK="${3:?Usage: ./push_experiment.sh /path/to/repo /path/to/staging_dir <track> <head_type> <slug>}"
HEAD_TYPE="${4:?Usage: ./push_experiment.sh /path/to/repo /path/to/staging_dir <track> <head_type> <slug>}"
SLUG="${5:?Usage: ./push_experiment.sh /path/to/repo /path/to/staging_dir <track> <head_type> <slug>}"

# Resolve to absolute paths before we cd into the repo, or relative paths break.
STAGING_DIR="$(cd "$STAGING_DIR" && pwd)"

branch="exp/${TRACK}/${HEAD_TYPE}/${SLUG}"

cd "$REPO_DIR"

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "Branch $branch already exists locally — checking it out instead of recreating."
  git checkout "$branch"
else
  git checkout main
  git pull --ff-only
  git checkout -b "$branch"
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

BASE="$STAGING_DIR"
copy_if_exists "$BASE/configs/$TRACK/$HEAD_TYPE/${SLUG}.yaml"                         "configs/$TRACK/$HEAD_TYPE/${SLUG}.yaml"
copy_if_exists "$BASE/models/$TRACK/$HEAD_TYPE/${SLUG}_model.py"                      "models/$TRACK/$HEAD_TYPE/${SLUG}_model.py"
copy_if_exists "$BASE/training/$TRACK/$HEAD_TYPE/${SLUG}_train.py"                    "training/$TRACK/$HEAD_TYPE/${SLUG}_train.py"
copy_if_exists "$BASE/evaluation/$TRACK/$HEAD_TYPE/${SLUG}/evaluate.py"               "evaluation/$TRACK/$HEAD_TYPE/${SLUG}/evaluate.py"
copy_if_exists "$BASE/evaluation/$TRACK/$HEAD_TYPE/${SLUG}/results.txt"               "evaluation/$TRACK/$HEAD_TYPE/${SLUG}/results.txt"
copy_if_exists "$BASE/experiments/$TRACK/$HEAD_TYPE/${SLUG}.md"                       "experiments/$TRACK/$HEAD_TYPE/${SLUG}.md"

if [ "${#copied[@]}" -eq 0 ]; then
  echo "Nothing found in $STAGING_DIR for track=$TRACK head_type=$HEAD_TYPE slug=$SLUG — check the names and try again."
  exit 1
fi

echo "Files staged for commit:"
printf '  %s\n' "${copied[@]}"

git add "${copied[@]}"
git commit -m "Add ${SLUG} (${TRACK}/${HEAD_TYPE}) experiment"
git push -u origin "$branch"

echo ""
echo "Pushed $branch. Open a PR for it now."
echo "Switch back to main with: git checkout main"
