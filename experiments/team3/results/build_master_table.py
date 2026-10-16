"""Merge reported and re-scored results into MASTER_RESULTS.csv and paper tables.

Inputs:
  * :mod:`reported_results` (numbers as reported by each track, with sources)
  * ``--scored``: CSV appended by ``python -m evaluation.common.score``
  * ``--llm-analysis``: analysis.json from ``llm_benchmark/analyze.py``
  * ``--llm-kaggle PUBLIC PRIVATE`` once the LLM submission is scored

Outputs (``--out-dir``): MASTER_RESULTS.csv, MAIN_RESULTS.md,
tab_main_results.tex, tab_llm.tex. Tables are never edited by hand.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from experiments.team3.results.reported_results import _row, all_reported

SCORED_TO_KEY = {
    "DualRoPE-CE (neural)": "t4_dualrope_ce",
    "DualRoPE-CE (v2)": "t4_dualrope_ce_v2",
    "DualRoPE-CRF (neural)": "t4_dualrope_crf",
    "DualRoPE-BoundaryCRF (neural)": "t4_dualrope_boundary_crf",
    "DualRoPE-CRF-RDrop (neural)": "t4_dualrope_crf_rdrop",
}
NEW_SCORED_ROWS = {
    "DualRoPE-CRF (v2)": dict(key="t4_dualrope_crf_v2", model="DziriFormer DualRoPE-CRF + lexical fallback"),
    "DualRoPE-BoundaryCRF (v2)": dict(key="t4_dualrope_boundary_crf_v2", model="DziriFormer DualRoPE-BoundaryCRF + lexical fallback"),
    "DualRoPE-CRF-RDrop (v2)": dict(key="t4_dualrope_crf_rdrop_v2", model="DziriFormer DualRoPE-CRF + R-Drop + lexical fallback"),
    "claude-opus-5-5 (zero_shot)": dict(key="llm_claude_zero", model="Claude Opus 5.5, zero-shot"),
    "claude-opus-5-5 (five_shot)": dict(key="llm_claude_five", model="Claude Opus 5.5, 5-shot"),
}
SCORED_FIELDS = {
    "accuracy": "dev_accuracy", "macro_f1": "dev_macro_f1", "wer": "dev_wer",
    "cer": "dev_cer", "der_star": "dev_der_star", "wer_star": "dev_wer_star",
}


def merge(repo: Path, scored_path: Path, llm_kaggle: Optional[List[float]]) -> List[Dict]:
    rows = all_reported(repo)
    by_key = {row["key"]: row for row in rows}
    with scored_path.open(encoding="utf-8") as stream:
        scored = list(csv.DictReader(stream))
    for entry in scored:
        if entry["split"] != "dev":
            continue  # dev_* columns only; test scores come from the leaderboard
        name = entry["model"]
        if name in SCORED_TO_KEY:
            row = by_key[SCORED_TO_KEY[name]]
        elif name in NEW_SCORED_ROWS:
            spec = NEW_SCORED_ROWS[name]
            is_llm = spec["key"].startswith("llm_")
            row = _row(
                family="LLM" if is_llm else "Transformer from scratch",
                slot="" if is_llm else "d",
                system_type="llm" if is_llm else entry["system_type"],
                main_table=is_llm,
                source="experiments/team3/llm_benchmark (outputs kept off-repo)" if is_llm else entry["source"],
                **spec,
            )
            rows.append(row)
            by_key[row["key"]] = row
        else:
            print(
                "warning: scored model {!r} has no table mapping and is not in MASTER_RESULTS.csv; "
                "add it to SCORED_TO_KEY or NEW_SCORED_ROWS".format(name),
                file=sys.stderr,
            )
            continue
        for scored_field, field in SCORED_FIELDS.items():
            row[field] = float(entry[scored_field])
        row["dev_scope"] = "full"
        row["evaluator"] = "shared scorer (evaluation/common)"
    if llm_kaggle:
        by_key["llm_claude_five"]["kaggle_public"] = llm_kaggle[0]
        by_key["llm_claude_five"]["kaggle_private"] = llm_kaggle[1]
    return rows


def pct(value, digits=2) -> str:
    return "--" if value in (None, "") else "{:.{}f}".format(100 * float(value), digits)


SCOPE_MARK = {"full": "", "slice20": "$^\\dagger$", "unknown": "$^\\ddagger$", "": ""}
MD_SCOPE_MARK = {"full": "", "slice20": " †", "unknown": " ‡", "": ""}
BLOCKS = [
    ("(a) Classical neural (BiLSTM)", lambda r: r["slot"] == "a" and r["system_type"] == "single"),
    ("(b) Arabic pretrained encoders", lambda r: r["slot"] == "b" and r["system_type"] == "single"),
    ("(c) Character-level pretrained (CANINE-S)", lambda r: r["slot"] == "c"),
    ("(d) Transformers trained from scratch", lambda r: r["slot"] == "d" and r["system_type"] == "single"),
    ("Systems: ensembles and lexical post-processing", lambda r: r["system_type"] in ("ensemble", "post-processed") and r["slot"]),
    ("References", lambda r: r["system_type"] in ("baseline", "llm")),
]


TYPE_LABEL = {
    "single": "single", "ensemble": "ensemble", "post-processed": "lexical",
    "baseline": "floor", "llm": "LLM",
}
MAIN_CAPTION = (
    "Baselines on DzTashkeel. Every fine-tuned system is trained on the Algerian corpus only; the LLM rows "
    "use prompting without any fine-tuning. Leaderboard columns give "
    "letter-level micro-F1 (\\%) on the hidden test split (public/private partitions); they are the only "
    "numbers computed identically for every system. Dev columns are as reported for each family. "
    "$^\\dagger$~Dev metrics on a held-out 20\\% of the dev split (3{,}228 letters). "
    "$^\\ddagger$~Family evaluator; dev scope not stated. $^\\S$~Leaderboard partition not stated. "
    "Unmarked dev numbers use the full dev split (15{,}897 letters)."
)
LLM_CAPTION = (
    "LLM versus fine-tuned systems on the full dev split (607 sentences, 15{,}897 letters), scored by the "
    "shared evaluator except the lookup floor, whose word-level figures come from the word-lexicon baseline "
    "of family (d)(ii) (unseen words counted as errors). DER$^*$/WER$^*$ exclude the final letter of "
    "multi-letter words. OOV and Ambig.\\ "
    "give WER on words unseen in training and on training words with more than one attested vocalization. "
    "All values in \\%."
)
STYLES = {
    "article": {
        "main_label": "tab:baselines", "llm_label": "tab:llm",
        "begin": ["\\begin{table*}[t]", "\\centering"],
        "tabular": lambda spec: ["\\small", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{" + spec.lower() + "}"],
        "end": ["\\end{tabular}}", "\\end{table*}"],
        "group": lambda n, title: "\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (n, title),
    },
    "cas": {
        "main_label": "tbl:baseline-results", "llm_label": "tbl:llm-comparison",
        "begin": ["\\begin{table}[width=\\linewidth,pos=htbp]"],
        "tabular": lambda spec: ["\\small", "\\begin{tabular*}{\\tblwidth}{@{}" + spec + "@{}}"],
        "end": ["\\end{tabular*}", "\\end{table}"],
        "group": lambda n, title: "\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\" % (n, title),
    },
}


def latex_text(text: str) -> str:
    """Escape the few characters that appear in model names."""
    return text.replace("&", "\\&").replace("\u00d7", "$\\times$")


def kaggle_cells(row) -> List[str]:
    if row["kaggle_unspecified"] is not None:
        return [pct(row["kaggle_unspecified"]) + "$^\\S$", "--"]
    return [pct(row["kaggle_public"]), pct(row["kaggle_private"])]


def main_table_rows(rows):
    for title, predicate in BLOCKS:
        members = [r for r in rows if r["main_table"] and predicate(r)]
        if members:
            yield title, members


def write_latex_main(rows, path: Path, style: str = "article") -> None:
    st = STYLES[style]
    best_private = max(r["kaggle_private"] or 0 for r in rows if r["main_table"])
    lines = ["% Generated by experiments/team3/results/build_master_table.py -- do not edit by hand."]
    lines += st["begin"]
    lines += ["\\caption{" + MAIN_CAPTION + "}", "\\label{" + st["main_label"] + "}"]
    lines += st["tabular"]("LLRRRR")
    lines += [
        "\\toprule",
        " & & \\multicolumn{2}{c}{Leaderboard (test)} & \\multicolumn{2}{c}{Dev} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
        "Model & Type & pub. & priv. & acc. & WER \\\\",
        "\\midrule",
    ]
    for title, members in main_table_rows(rows):
        lines.append(st["group"](6, title))
        for r in members:
            public, private = kaggle_cells(r)
            if r["kaggle_private"] == best_private:
                private = "\\textbf{" + private + "}"
            mark = SCOPE_MARK[r["dev_scope"]]
            acc = pct(r["dev_accuracy"]) + (mark if r["dev_accuracy"] is not None else "")
            wer = pct(r["dev_wer"]) + (mark if r["dev_wer"] is not None else "")
            lines.append(
                "\\quad {} & {} & {} & {} & {} & {} \\\\".format(
                    latex_text(r["model"]), TYPE_LABEL[r["system_type"]], public, private, acc, wer
                )
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += st["end"] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_main(rows, path: Path) -> None:
    lines = [
        "# Main results (generated -- do not edit)\n",
        "Kaggle = letter micro-F1 (%) on the hidden test split. Dev as reported. "
        "† 20% dev slice. ‡ family evaluator, scope not stated. § leaderboard partition not stated.\n",
        "| Model | Type | Kaggle pub. | Kaggle priv. | Dev acc. | Dev WER | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for title, members in main_table_rows(rows):
        lines.append("| **{}** | | | | | | |".format(title))
        for r in members:
            public, private = kaggle_cells(r)
            mark = MD_SCOPE_MARK[r["dev_scope"]]
            lines.append(
                "| {} | {} | {} | {} | {}{} | {}{} | {} |".format(
                    r["model"], TYPE_LABEL[r["system_type"]], public.replace("$^\\S$", " §"), private,
                    pct(r["dev_accuracy"]), mark if r["dev_accuracy"] is not None else "",
                    pct(r["dev_wer"]), mark if r["dev_wer"] is not None else "", r["source"],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_llm(rows, analysis: Dict, path: Path, style: str = "article") -> None:
    st = STYLES[style]
    by_key = {r["key"]: r for r in rows}
    systems = [
        ("llm_claude_zero", "llm_zero", "Claude Opus 5.5, zero-shot"),
        ("llm_claude_five", "llm_five", "Claude Opus 5.5, 5-shot"),
        ("t4_dualrope_crf_rdrop", "t4_neural", "Best single fine-tuned (DziriFormer, neural)"),
        ("t4_dualrope_crf_rdrop_v2", "t4_v2", "Best fine-tuned system (DziriFormer + lexical fallback)"),
    ]
    lines = ["% Generated by experiments/team3/results/build_master_table.py -- do not edit by hand."]
    lines += st["begin"]
    lines += ["\\caption{" + LLM_CAPTION + "}", "\\label{" + st["llm_label"] + "}"]
    lines += st["tabular"]("LRRRRRRR")
    lines += [
        "\\toprule",
        "System & Acc. & DER$^*$ & WER & WER$^*$ & CER & WER OOV & WER Ambig. \\\\",
        "\\midrule",
    ]
    for key, analysis_name, label in systems:
        if key not in by_key or analysis_name not in analysis["systems"]:
            print("tab_llm: skipping {} (not scored or not analyzed)".format(label))
            continue
        r = by_key[key]
        buckets = analysis["systems"][analysis_name]["bucket_wer"]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                label, pct(r["dev_accuracy"]), pct(r["dev_der_star"]), pct(r["dev_wer"]),
                pct(r["dev_wer_star"]), pct(r["dev_cer"]),
                pct(buckets["oov"]["wer"]), pct(buckets["seen_ambiguous"]["wer"]),
            )
        )
    lines += [
        "Lookup floor (most frequent vocalization) & -- & -- & 26.07 & -- & -- & 100.00 & 24.61 \\\\",
        "\\bottomrule",
    ] + st["end"] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--llm-analysis", type=Path, required=True)
    parser.add_argument("--llm-kaggle", type=float, nargs=2, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--style", choices=sorted(STYLES), default="article",
        help="table wrappers: plain article, or Elsevier CAS (tabular* with \\tblwidth, tbl: labels)",
    )
    args = parser.parse_args()

    rows = merge(args.repo, args.scored, args.llm_kaggle)
    analysis = json.loads(args.llm_analysis.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "MASTER_RESULTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_main(rows, args.out_dir / "MAIN_RESULTS.md")
    write_latex_main(rows, args.out_dir / "tab_main_results.tex", args.style)
    write_latex_llm(rows, analysis, args.out_dir / "tab_llm.tex", args.style)
    print("{} rows -> {}".format(len(rows), args.out_dir))


if __name__ == "__main__":
    main()
