#!/usr/bin/env bash
#
# organize_by_type.sh — Phase 1: split each experiment into the repo's own
# folders, following the naming conventions in each folder's own README.md.
#
#   configs/<track>/<head_type>/<strategy>_<slug>.yaml
#     - run_id: set to the unique <slug> (traceability)
#     - model_registry_key: the actual MODEL_REGISTRY key used to select
#       the backbone (may repeat across configs on purpose, e.g. two runs
#       of "camelbert_da" — run_id disambiguates them, this doesn't need to)
#   models/<track>/<head_type>/<slug>_model.py
#     -> consolidate.py later collapses to <head_type>_model.py if identical
#   training/<track>/<head_type>/<slug>_finetune.py
#     -> consolidate.py later collapses to finetune_<head_type>.py if identical
#   evaluation/<track>/<head_type>/<slug>_evaluate.py
#     -> consolidate.py later collapses to evaluate_<head_type>.py if identical
#   evaluation/<track>/<head_type>/report_<strategy>_<slug>.md
#     - DER (1 - micro-F1) is the primary metric; no competition/leaderboard
#       language anywhere — this is a benchmark report, not a submission record.
#     - "validation set" language, not "dev/test" — this is your held-out
#       split of labeled training data, not a blind competition test set.
#   experiments/<track>/<head_type>/<strategy>_overview.md
#   staging/scores.csv — slug,track,head_type,strategy,micro_f1,der — used by
#     consolidate.py (to pick the best-scoring default --active-model) and
#     by results_summary.md below. Not meant to be committed to the repo.
#
# USAGE:
#   ./organize_by_type.sh /path/to/raw_experiments /path/to/staging_dir

set -euo pipefail

RAW_DIR="${1:?Usage: ./organize_by_type.sh /path/to/raw_experiments /path/to/staging_dir}"
STAGING_DIR="${2:?Usage: ./organize_by_type.sh /path/to/raw_experiments /path/to/staging_dir}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAP_FILE="$SCRIPT_DIR/experiments_map.csv"

if [ ! -f "$MAP_FILE" ]; then
  echo "Missing $MAP_FILE"
  exit 1
fi

mkdir -p "$STAGING_DIR/configs" "$STAGING_DIR/training" "$STAGING_DIR/models" "$STAGING_DIR/evaluation" "$STAGING_DIR/experiments"
echo "model_slug,track,head_type,strategy,micro_f1,der" > "$STAGING_DIR/scores.csv"

tail -n +2 "$MAP_FILE" | while IFS=',' read -r folder_name strategy model_slug track head_type; do
  folder_name="$(echo "$folder_name" | xargs)"
  strategy="$(echo "$strategy" | xargs)"
  model_slug="$(echo "$model_slug" | xargs)"
  track="$(echo "$track" | xargs)"
  head_type="$(echo "$head_type" | xargs)"
  [ -z "$folder_name" ] && continue

  src="${RAW_DIR}/${folder_name}"
  if [ ! -d "$src" ]; then
    echo "SKIP: $src not found"
    continue
  fi
  echo "=== Splitting $folder_name ($model_slug, $track/$head_type, $strategy) ==="

  # 1. config.json -> configs/<track>/<head_type>/<strategy>_<slug>.yaml
  if [ -f "$src/config.json" ]; then
    mkdir -p "$STAGING_DIR/configs/$track/$head_type"
    python3 -c "
import json, yaml
with open('$src/config.json', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['model_registry_key'] = cfg.get('run_id')  # the actual MODEL_REGISTRY key -- may repeat across runs on purpose
cfg['run_id'] = '$model_slug'                  # unique per run -- used for checkpoint/export paths, must not collide
with open('$STAGING_DIR/configs/$track/$head_type/${strategy}_${model_slug}.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
"
  fi

  # 2. the notebook -> models/, training/, evaluation/ (split by section header, flat naming)
  notebook_inside=$(find "$src" -maxdepth 1 -iname "*.ipynb" 2>/dev/null | head -n1)
  if [ -n "$notebook_inside" ]; then
    python3 "$SCRIPT_DIR/split_notebook.py" "$notebook_inside" "$model_slug" "$track/$head_type" "$STAGING_DIR"
    train_flat="$STAGING_DIR/training/$track/$head_type/${model_slug}_train.py"
    if [ -f "$train_flat" ]; then
      mv "$train_flat" "$STAGING_DIR/training/$track/$head_type/${model_slug}_finetune.py"
    fi
  else
    echo "  (no .ipynb found directly inside $src — check it by hand)"
  fi

  # 3. results report -> evaluation/<track>/<head_type>/report_<strategy>_<slug>.md
  #    Primary metric: DER (= 1 - micro-F1). No competition/leaderboard language.
  mkdir -p "$STAGING_DIR/evaluation/$track/$head_type"
  python3 - "$src" "$STAGING_DIR" "$model_slug" "$strategy" "$track" "$head_type" <<'PYEOF'
import json, sys, os, csv

src, staging, slug, strategy, track, head_type = sys.argv[1:7]
eval_path = os.path.join(src, "dev_test_evaluation.json")

out_dir = os.path.join(staging, "evaluation", track, head_type)
os.makedirs(out_dir, exist_ok=True)

lines = [f"# Evaluation Report — {slug}\n",
         f"**Track:** {track} | **Head:** {head_type} | **Strategy:** {strategy}\n"]

micro_f1, der = None, None

if os.path.exists(eval_path):
    with open(eval_path, encoding='utf-8') as f:
        ev = json.load(f)
    cr = ev.get("classification_report")
    micro_f1 = ev.get('micro_f1')

    if micro_f1 is not None:
        der = 1 - micro_f1
        lines.append(f"- **DER (Diacritic Error Rate): {der:.4f}**  "
                      f"(= 1 - micro-F1; micro-F1 = {micro_f1:.4f})")

    if cr and "macro avg" in cr:
        lines.append(f"- Macro-F1, all 16 classes (matches table below): **{cr['macro avg']['f1-score']:.4f}**")
    present_macro = ev.get('macro_f1_present_classes')
    n_present = ev.get('n_present_classes')
    if present_macro is None and 'macro_f1' in ev:
        present_macro = ev['macro_f1']
    if present_macro is not None:
        suffix = f", {n_present} classes" if n_present is not None else ""
        lines.append(f"- Macro-F1, classes with support only{suffix} (excludes 0-support classes, "
                      f"NOT comparable to the line above): **{present_macro:.4f}**")

    lines.append(f"- Characters evaluated (validation set): {ev.get('n_chars_evaluated', 'n/a')}\n")

    if cr:
        lines.append("## Per-class metrics (validation set)\n")
        lines.append("| Class | Precision | Recall | F1-score | Support |")
        lines.append("|---|---|---|---|---|")
        for cls, m in cr.items():
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']:.0f} |")
        if "accuracy" in cr:
            support = cr.get("weighted avg", {}).get("support", "")
            lines.append(f"| **accuracy** | | | {cr['accuracy']:.4f} | {support:.0f} |")
        for cls in ("macro avg", "weighted avg"):
            if cls in cr:
                m = cr[cls]
                lines.append(f"| **{cls}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']:.0f} |")
else:
    lines.append("_(No dev_test_evaluation.json found for this experiment.)_")

with open(os.path.join(out_dir, f"report_{strategy}_{slug}.md"), "w", encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")

# Record for scores.csv (used by consolidate.py + results_summary.md)
scores_path = os.path.join(staging, "scores.csv")
with open(scores_path, "a", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([slug, track, head_type, strategy,
                f"{micro_f1:.6f}" if micro_f1 is not None else "",
                f"{der:.6f}" if der is not None else ""])
PYEOF

done

echo ""
echo "=== Consolidating duplicated model/eval/train scripts within each track/head_type group ==="
python3 "$SCRIPT_DIR/consolidate.py" "$STAGING_DIR" "$MAP_FILE"

echo ""
echo "=== Building per-group overview files in experiments/ ==="
python3 "$SCRIPT_DIR/build_overviews.py" "$STAGING_DIR" "$MAP_FILE"

# Combined results summary, ranked by DER (lower is better)
python3 - "$STAGING_DIR" <<'PYEOF'
import csv, sys, os
staging_dir = sys.argv[1]
rows = []
with open(os.path.join(staging_dir, "scores.csv"), encoding='utf-8') as f:
    for row in csv.DictReader(f):
        try:
            der = float(row["der"])
        except (KeyError, ValueError):
            der = float('inf')
        rows.append((der, row))
rows.sort(key=lambda x: x[0])
lines = ["# Results Summary\n", "Ranked by DER (Diacritic Error Rate — lower is better).\n",
         "| Rank | Model | Track | Head | Strategy | Micro-F1 | DER |", "|---|---|---|---|---|---|---|"]
for i, (der, row) in enumerate(rows, 1):
    der_s = f"{der:.4f}" if der != float('inf') else "n/a"
    lines.append(f"| {i} | {row['model_slug']} | {row['track']} | {row['head_type']} | {row['strategy']} | {row.get('micro_f1','n/a')} | {der_s} |")
with open(os.path.join(staging_dir, "results_summary.md"), "w", encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {os.path.join(staging_dir, 'results_summary.md')}")
PYEOF

echo ""
echo "Done. Review:"
echo "  $STAGING_DIR/configs/<track>/<head_type>/"
echo "  $STAGING_DIR/models/<track>/<head_type>/"
echo "  $STAGING_DIR/training/<track>/<head_type>/"
echo "  $STAGING_DIR/evaluation/<track>/<head_type>/"
echo "  $STAGING_DIR/experiments/<track>/<head_type>/  (one overview.md per group)"
echo "  $STAGING_DIR/results_summary.md"
echo ""
echo "IMPORTANT: open the generated models/*.py and evaluation/*_evaluate.py files and"
echo "check their imports before pushing — the split is by notebook section, not by"
echo "tracing actual code dependencies. Imports are auto-propagated now, but cross-file"
echo "globals (CFG, BACKBONE_NAME, CHAR2ID, etc.) are NOT — that needs a real refactor."
