"""
build_overviews.py — builds one overview.md per (track, head_type, strategy)
group in experiments/, per experiments/README.md: this folder is for
"experiment plans, strategy-level run definitions, and entry points," not
per-run logs (those live in evaluation/report_*.md instead).

Ranks by DER (Diacritic Error Rate = 1 - micro-F1), read from
staging/scores.csv (written by organize_by_type.sh). No competition/
leaderboard language -- this is a benchmark, not a submission record.

USAGE:
    python3 build_overviews.py <staging_dir> <experiments_map.csv>
"""
import csv, sys, os

def load_scores(staging):
    scores_path = os.path.join(staging, "scores.csv")
    scores = {}
    if not os.path.exists(scores_path):
        return scores
    with open(scores_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            scores[row["model_slug"]] = row
    return scores

def main(staging, map_file):
    scores = load_scores(staging)

    groups = {}
    with open(map_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row["track"].strip(), row["head_type"].strip(), row["strategy"].strip())
            groups.setdefault(key, []).append(row)

    for (track, head_type, strategy), rows in groups.items():
        def der_of(r):
            s = scores.get(r["model_slug"], {})
            try:
                return float(s.get("der"))
            except (TypeError, ValueError):
                return float('inf')
        rows_sorted = sorted(rows, key=der_of)

        lines = [f"# {strategy} -- {track} / {head_type}\n"]
        lines.append(f"{len(rows)} experiment(s) in this group, ranked by DER "
                      f"(Diacritic Error Rate -- lower is better). Shared code:")
        lines.append(f"- Model: `models/{track}/{head_type}/{head_type}_model.py`")
        lines.append(f"- Fine-tuning script: `training/{track}/{head_type}/finetune_{head_type}.py`")
        lines.append(f"- Evaluation code: `evaluation/{track}/{head_type}/evaluate_{head_type}.py`\n")
        lines.append("| Model | Config | Report | Micro-F1 | DER |")
        lines.append("|---|---|---|---|---|")
        for r in rows_sorted:
            slug = r["model_slug"]
            cfg = f"`configs/{track}/{head_type}/{strategy}_{slug}.yaml`"
            report = f"`evaluation/{track}/{head_type}/report_{strategy}_{slug}.md`"
            s = scores.get(slug, {})
            micro_f1 = s.get("micro_f1", "")
            der = s.get("der", "")
            micro_f1_s = f"{float(micro_f1):.4f}" if micro_f1 else "n/a"
            der_s = f"{float(der):.4f}" if der else "n/a"
            lines.append(f"| {slug} | {cfg} | {report} | {micro_f1_s} | {der_s} |")

        out_dir = os.path.join(staging, "experiments", track, head_type)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{strategy}_overview.md"), "w", encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {os.path.join(out_dir, f'{strategy}_overview.md')}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
