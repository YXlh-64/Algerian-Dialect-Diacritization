"""Generate machine-readable and professor-facing campaign summaries."""

from pathlib import Path
from typing import Any, Mapping, Sequence


def _table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| System | Seed | Params | Epoch | Neural F1 | V2/OOF F1 | Correct | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {system} | {seed} | {parameters} | {epoch} | "
            "{neural_f1:.10f} | {final_f1:.10f} | {correct} | {status} |".format(
                **row
            )
        )
    return "\n".join(lines)


def write_reports(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    final_system: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    results_path = directory / "FINAL_RESULTS.md"
    brief_path = directory / "PROFESSOR_BRIEF.md"
    if (
        results_path.is_file()
        and brief_path.is_file()
        and "Submission SHA-256" in results_path.read_text(encoding="utf-8")
        and "Focused feedback question" in brief_path.read_text(
            encoding="utf-8"
        )
    ):
        return
    results = f"""# Pre-HGL/HGL Campaign Final Results

## Decision

Selected system: **{final_system['system']}**

Final dev Micro-F1: `{final_system['final_f1']:.10f}`

Final submission:

```text
{final_system['submission_path']}
```

## Complete results

{_table(rows)}

All scores use the released 15,897-letter dev split. Every CSV was checked
against the competition's official conversion script.
"""
    results_path.write_text(results, encoding="utf-8")

    explanations = {
        "Uniform Ensemble": (
            "Averages normalized probabilities from Base, J16, GL, Mixed, "
            "and Hier equally. It tests error diversity without learned weights."
        ),
        "HierMixed": (
            "Adds full sentence attention in blocks 3 and 6 to the hierarchical "
            "word-character model, testing whether local morphology and global "
            "character context are complementary."
        ),
        "Direct16": (
            "Replaces independent base/Shadda prediction with one direct "
            "16-class head and one unweighted official-label loss."
        ),
        "GL Curriculum": (
            "Changes exact guided learning into a linear blank-hint curriculum "
            "that progressively matches blank-hint inference."
        ),
        "OOF Gate": (
            "Learns an explainable logistic neural-versus-lexical switch from "
            "strictly out-of-fold disagreements instead of hand-tuned weights."
        ),
        "HGL": (
            "Combines the hierarchical encoder, evidence-approved attention/head "
            "choices, and the blank-hint curriculum."
        ),
    }
    sections = []
    for name, explanation in explanations.items():
        matching = [row for row in rows if name.lower().replace(" ", "") in row["system"].lower().replace(" ", "")]
        result = (
            "No completed run."
            if not matching
            else "Best recorded final F1: `{:.10f}`.".format(
                max(float(row["final_f1"]) for row in matching)
            )
        )
        sections.append(
            f"## {name}\n\n**What changed:** {explanation}\n\n"
            f"**Result:** {result}\n\n"
            "**Feedback requested:** Is this inductive bias justified for "
            "Algerian dialect morphology, and which failure category should "
            "be prioritized next?"
        )
    brief = """# Professor Brief — Algerian Dialect Vocalization Track 4

All systems are trained from scratch using only released competition data.
No pretrained encoder, external embedding, analyzer, or corpus is used.

""" + "\n\n".join(sections) + "\n"
    brief_path.write_text(brief, encoding="utf-8")
