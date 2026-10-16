"""Registry of results *as reported* by each track, with their sources.

Decision for the paper (Team 3, Oct 2026): §6 quotes the numbers each track
reported. The Kaggle public/private micro-F1 is the cross-track comparison
because every track was scored on the same hidden test split with the same
metric. Dev numbers are quoted with their evaluation scope:

    full     released dev split, 607 sentences / 15,897 letters
    slice20  Track 3's held-out 20% of dev (about 122 sentences / 3,228 letters)
    unknown  scope not stated by the source

Rows produced by the shared scorer (``evaluation/common``) are merged in by
``build_master_table.py`` and override the dev columns of a matching ``key``.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, List

SOUNDOUS = "origin/Track1/Soundous/BiLSTM-based-tagger-family:documentation/track1/soundous/README.md"
THABET = "origin/Track1/Thabet/P2-Strategy-A:evaluation/track1/bilstm_cnn_crf/report_strategy_a_p2_ensemble_09483.md"
ROUMAISSA_S6 = "section6_Roumaissa.pdf, Table 1 (Track 2 draft for §6)"
T2_OVERVIEW = "experiments/track2/canine_s_model/strategy_a_overview.md"
LYES = "Lyes local README.md (Kaggle submissions table) + outputs/paper_metrics_v2"
SOUHA = "documentation/track4/AbidatSouha/results.md"
ROUMAISSA_T4 = "documentation/track4/SmailRoumaissa/README.md"


def _row(**values) -> Dict:
    base = {
        "key": "",
        "family": "",
        "slot": "",
        "model": "",
        "system_type": "single",
        "kaggle_public": None,
        "kaggle_private": None,
        "kaggle_unspecified": None,
        "dev_accuracy": None,
        "dev_macro_f1": None,
        "dev_wer": None,
        "dev_cer": None,
        "dev_der_star": None,
        "dev_wer_star": None,
        "dev_scope": "",
        "evaluator": "",
        "main_table": False,
        "source": "",
        "note": "",
    }
    base.update(values)
    return base


def track1_rows() -> List[Dict]:
    soundous = [
        ("t1_bilstm_cnn", "BiLSTM-CNN", 0.9499, 0.1930, True, "single"),
        ("t1_bilstm_crf", "BiLSTM-CRF", 0.9455, 0.2059, True, "single"),
        ("t1_bilstm_cnn_crf", "BiLSTM-CNN-CRF", 0.9510, 0.1912, True, "single"),
        ("t1_focal", "BiLSTM-CNN-CRF + focal reweighting", 0.9499, 0.1948, False, "single"),
        ("t1_multitask", "BiLSTM-CNN-CRF + has-diacritic head", 0.9505, 0.1904, False, "single"),
        ("t1_attention", "BiLSTM-CNN-CRF + self-attention", 0.9514, 0.1915, False, "single"),
        ("t1_consistency", "BiLSTM-CNN-CRF + consistency augmentation", 0.9512, 0.1907, False, "single"),
        ("t1_swa", "BiLSTM-CNN-CRF + SWA", 0.9501, 0.1930, False, "single"),
        ("t1_curriculum", "BiLSTM-CNN-CRF + length curriculum", 0.9514, 0.1881, False, "single"),
        ("t1_tta", "BiLSTM-CNN-CRF + multi-offset TTA", 0.9510, 0.1912, False, "post-processed"),
        ("t1_seed_ensemble", "BiLSTM-CNN-CRF 3-seed ensemble", 0.9526, 0.1855, False, "ensemble"),
    ]
    rows = [
        _row(
            key=key, family="Classical neural", slot="a", model=model,
            system_type=kind, dev_accuracy=acc, dev_wer=wer, dev_scope="unknown",
            evaluator="Track 1 (label-aligned WER)", main_table=main, source=SOUNDOUS,
            note="Dev accuracy conflicts with Thabet's full-dev 0.94395 for the ensemble; denominators to confirm.",
        )
        for key, model, acc, wer, main, kind in soundous
    ]
    rows.append(
        _row(
            key="t1_p2_ensemble", family="Classical neural", slot="a",
            model="BiLSTM-CNN-CRF ensemble", system_type="ensemble",
            kaggle_public=0.95048, kaggle_private=0.94829, dev_accuracy=0.94395,
            dev_macro_f1=0.58610, dev_scope="full", evaluator="Track 1 (Thabet)",
            main_table=True, source=THABET,
        )
    )
    return rows


def track2_rows() -> List[Dict]:
    return [
        _row(
            key="t2_canine_s", family="Character-level pretrained", slot="c",
            model="CANINE-S", kaggle_public=0.93880, kaggle_private=0.93369,
            dev_accuracy=0.9336, dev_macro_f1=0.4559, dev_scope="full",
            evaluator="Track 2", main_table=True, source=ROUMAISSA_S6,
            note="Repo overview ({}) reports dev 0.9452 / private 0.93235 instead; "
            "draft dev accuracy equals its Kaggle private score to 4 decimals.".format(T2_OVERVIEW),
        ),
        _row(
            key="t2_canine_s_twohead", family="Character-level pretrained", slot="c",
            model="CANINE-S, factorized head (Shadda × vowel)", kaggle_public=0.94427,
            kaggle_private=0.94062, dev_accuracy=0.9406, dev_macro_f1=0.5281,
            dev_scope="full", evaluator="Track 2", main_table=True, source=ROUMAISSA_S6,
            note="Draft dev accuracy equals its Kaggle private score to 4 decimals; repo overview gives macro-F1 0.7040.",
        ),
    ]


def _git_show(repo: Path, spec: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", spec], check=True, capture_output=True, text=True
    ).stdout


def track3_rows(repo: Path) -> List[Dict]:
    leaderboard = _git_show(repo, "HEAD:experiments/leaderboard.md")
    kaggle = {}
    for line in leaderboard.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7 and cells[2] == "track3":
            kaggle[(cells[1], cells[3])] = (float(cells[5]), float(cells[6]))
    pretty = {
        "arabert_v02": "AraBERTv02", "camelbert_da": "CAMeLBERT-DA", "camelbert_mix": "CAMeLBERT-Mix",
        "marbert": "MARBERT", "marbertv2": "MARBERTv2", "dziribert": "DziriBERT", "alclam": "AlcLaM",
    }
    main = {
        ("arabert_v02_09517", "bilstm_crf_head"), ("dziribert_09511", "bilstm_crf_head"),
        ("marbert_09498", "bilstm_crf_head"), ("camelbert_da_09483", "bilstm_crf_head"),
        ("camelbert_mix_09470", "bilstm_crf_head"), ("arabert_v02_09157", "linear_head"),
    }
    rows = []
    for (name, head), (public, private) in sorted(kaggle.items()):
        report = _git_show(
            repo, "HEAD:evaluation/track3/{}/report_strategy_a_{}.md".format(head, name)
        )
        micro = float(re.search(r"micro-F1 = ([0-9.]+)", report).group(1))
        macro = float(re.search(r"Macro-F1, all 16 classes[^*]*\*\*([0-9.]+)\*\*", report).group(1))
        letters = int(re.search(r"Characters evaluated \(validation set\): (\d+)", report).group(1))
        backbone = pretty[name.rsplit("_", 1)[0]]
        rows.append(
            _row(
                key="t3_{}_{}".format(name, head), family="Arabic pretrained encoder", slot="b",
                model="{} + {}".format(backbone, "BiLSTM-CRF head" if head == "bilstm_crf_head" else "linear head"),
                kaggle_public=public, kaggle_private=private, dev_accuracy=micro,
                dev_macro_f1=macro, dev_scope="slice20" if letters < 15897 else "full",
                evaluator="Track 3", main_table=(name, head) in main,
                source="evaluation/track3/{}/report_strategy_a_{}.md + experiments/leaderboard.md".format(head, name),
                note="Dev scored on {} letters (held-out 20% of dev).".format(letters),
            )
        )
    rows.append(
        _row(
            key="t3_camelbert_mix_factorized", family="Arabic pretrained encoder", slot="b",
            model="CAMeLBERT-Mix + factorized BiLSTM-CRF head", kaggle_public=0.94646,
            kaggle_private=0.93977, dev_accuracy=0.9421, dev_wer=0.1869, dev_cer=0.0313,
            dev_der_star=0.0698, dev_wer_star=0.1697, dev_scope="unknown", evaluator="Track 3",
            source="experiments/track3/additional_experiments.md",
        )
    )
    return rows


def track4_reported_rows() -> List[Dict]:
    return [
        _row(key="t4_dualrope_ce", family="Transformer from scratch", slot="d",
             model="DziriFormer DualRoPE-CE", kaggle_public=0.93466, kaggle_private=0.92541,
             main_table=True, source=LYES),
        _row(key="t4_dualrope_crf", family="Transformer from scratch", slot="d",
             model="DziriFormer DualRoPE-CRF", main_table=True, source=LYES),
        _row(key="t4_dualrope_boundary_crf", family="Transformer from scratch", slot="d",
             model="DziriFormer DualRoPE-BoundaryCRF", main_table=True, source=LYES),
        _row(key="t4_dualrope_crf_rdrop", family="Transformer from scratch", slot="d",
             model="DziriFormer DualRoPE-CRF + emission R-Drop", source=LYES),
        _row(key="t4_dualrope_ce_v2", family="Transformer from scratch", slot="d",
             model="DziriFormer DualRoPE-CE + lexical fallback", system_type="post-processed",
             kaggle_public=0.94634, source=LYES),
        _row(key="t4_dziriformer_ensemble_v2", family="Transformer from scratch", slot="d",
             model="DziriFormer CRF ensemble + lexical fallback", system_type="ensemble",
             kaggle_public=0.94841, kaggle_private=0.94257, dev_accuracy=14977 / 15897,
             dev_scope="full", evaluator="Track 4 paper metrics", main_table=True, source=LYES,
             note="Dev = 14,977 / 15,897 letters (equal-weight production ensemble)."),
        _row(key="t4_souha_tcc", family="Transformer from scratch", slot="d",
             model="Transformer-CNN-CRF (ii), word features", kaggle_unspecified=0.93977,
             dev_accuracy=0.9343, dev_wer=0.2023, dev_scope="full", evaluator="Track 4 (Souha)",
             main_table=True, source=SOUHA,
             note="Leaderboard split (public/private) not stated; macro-F1 81.91 is over 9 live classes."),
        _row(key="t4_roumaissa_tcc", family="Transformer from scratch", slot="d",
             model="Transformer-CNN-CRF (iii), relative positions", dev_accuracy=0.93716, dev_scope="full",
             evaluator="Track 4 (Roumaissa)", main_table=True, source=ROUMAISSA_T4,
             note="Calibrated neural-only dev score."),
        _row(key="t4_roumaissa_tcc_lex", family="Transformer from scratch", slot="d",
             model="Transformer-CNN-CRF (iii) + lexical fusion",
             system_type="post-processed", kaggle_public=0.94610, kaggle_private=0.94427,
             dev_accuracy=0.94213, dev_scope="full", evaluator="Track 4 (Roumaissa)",
             main_table=True, source=ROUMAISSA_T4,
             note="Fusion gate tuned on dev (45 combos), so dev is optimistic."),
    ]


def reference_rows() -> List[Dict]:
    return [
        _row(key="floor_lookup", family="Reference", model="Most-frequent vocalization lookup",
             system_type="baseline", dev_wer=0.2607, dev_scope="full",
             evaluator="Track 4 (Souha)", main_table=True, source=SOUHA),
    ]


def all_reported(repo: Path) -> List[Dict]:
    return (
        track1_rows() + track3_rows(repo) + track2_rows()
        + track4_reported_rows() + reference_rows()
    )
