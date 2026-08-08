"""
consolidate.py — after organize_by_type.sh has split all experiments,
detect duplicated model.py / evaluate.py / train.py within each
(track, head_type) group and collapse them into one shared file, since
these notebooks are built on a MODEL_REGISTRY/ACTIVE_MODEL pattern where
only the config actually differs per model.

Safe by design: for any group, if the files AREN'T identical (beyond the
auto-generated header and, for the training script, the ACTIVE_MODEL
line), nothing is collapsed for that group and a warning is printed.

USAGE:
    python3 consolidate.py <staging_dir> <experiments_map.csv>
"""
import csv, sys, os, re

HEADER_PATTERN = re.compile(
    r'^# Auto-split from .*\n# NOTE: shared imports/setup live in .*\n(# Sanity-check imports here before relying on this file standalone\.\n)?',
    re.MULTILINE
)

ACTIVE_MODEL_LINE = re.compile(r'^ACTIVE_MODEL:\s*str\s*=\s*"[^"]*"\s*$', re.MULTILINE)

# Cosmetic/noise patterns that show up inconsistently across copy-pasted
# notebooks but aren't real code differences — stripped before comparing,
# and stripped from the final shared output too where they're actively
# broken (shell magics aren't valid Python outside a notebook).
KAGGLE_DOWNLOAD_CELL = re.compile(
    r'\n*!pip install kaggle\n!kaggle kernels output [^\n]*\n*', re.MULTILINE
)


def strip_noise(text):
    return KAGGLE_DOWNLOAD_CELL.sub('\n', text)


# The export_include_weights block is a real, config-gated feature that's
# missing from some copies by accident (the config already has
# export_include_weights: true in every group member). Stripped only for
# the *comparison* — the richest file (the one that has it) is still used
# as the template, so the shared output keeps the feature.
EXPORT_WEIGHTS_BLOCK = re.compile(
    r'\n {4}if cfg\.export_include_weights:\n(?: {8}.*\n)+',
)
EXPORT_WEIGHTS_LOGLINE = re.compile(
    r'\n\s*f"Weights included: \{cfg\.export_include_weights\}\\n"'
)


def strip_optional_blocks(text):
    text = EXPORT_WEIGHTS_BLOCK.sub('\n', text)
    text = EXPORT_WEIGHTS_LOGLINE.sub('', text)
    return text


def compare_key(text, mask_active_model=False):
    """Aggressive normalization used ONLY to decide whether two files are
    the same — eliminates blank-line-count differences entirely, since
    those are never semantically meaningful in Python."""
    text = text.replace('\r\n', '\n')
    text = HEADER_PATTERN.sub('', text, count=1)
    text = strip_noise(text)
    text = strip_optional_blocks(text)
    if mask_active_model:
        text = ACTIVE_MODEL_LINE.sub('ACTIVE_MODEL: str = "<MASKED>"', text)
    text = re.sub(r'\n[ \t]*\n+', '\n', text)  # collapse ALL blank lines away
    text = re.sub(r'[ \t]+\n', '\n', text)     # trailing whitespace per line
    return text.strip()


def clean_output(text):
    """Light cleanup used for the file actually written to disk — strips
    the header and known noise, but preserves the original author's blank
    lines and formatting."""
    text = text.replace('\r\n', '\n')
    text = HEADER_PATTERN.sub('', text, count=1)
    text = strip_noise(text)
    return text.strip()


def normalize(text, mask_active_model=False):
    # Back-compat alias for comparison use.
    return compare_key(text, mask_active_model=mask_active_model)


def cluster_by_content(items):
    """Group (key, normalized_content) pairs into clusters of identical
    content. Returns list of (content, [keys]) sorted by cluster size desc."""
    clusters = {}
    for key, content in items:
        clusters.setdefault(content, []).append(key)
    return sorted(clusters.items(), key=lambda kv: -len(kv[1]))


def collapse_simple(staging, track, head_type, slugs, path_fn, shared_path, label):
    """For model.py / evaluate.py: cluster by identical content (after
    stripping header/noise), collapse the majority cluster into one shared
    file, and leave any genuine outliers as their own separate files."""
    paths = {s: path_fn(staging, track, head_type, s) for s in slugs}
    missing = {s: p for s, p in paths.items() if not os.path.exists(p)}
    paths = {s: p for s, p in paths.items() if os.path.exists(p)}
    if missing:
        print(f"  NOTE for {track}/{head_type} ({label}): {len(missing)} expected file(s) not found on disk:")
        for s, p in missing.items():
            print(f"    missing: {p}")
    if len(paths) < 2:
        print(f"  SKIP {label} collapse for {track}/{head_type}: only found {len(paths)} of {len(slugs)} expected files.")
        return False

    items = [(s, compare_key(open(p, encoding='utf-8').read())) for s, p in paths.items()]
    clusters = cluster_by_content(items)

    if len(clusters) == 1:
        print(f"  Collapsed {len(paths)} {label} files -> {shared_path}")
    else:
        outliers = [s for _, keys in clusters[1:] for s in keys]
        print(f"  {label} for {track}/{head_type}: {len(clusters[0][1])} of {len(paths)} are identical "
              f"and collapsed; {len(outliers)} genuinely differ and were LEFT SEPARATE: {', '.join(outliers)}")

    _, majority_slugs = clusters[0]
    if len(majority_slugs) < 2:
        print(f"  SKIP {label} collapse for {track}/{head_type}: no two files actually match — all differ.")
        return False

    representative_content = clean_output(open(paths[majority_slugs[0]], encoding='utf-8').read())

    os.makedirs(os.path.dirname(shared_path), exist_ok=True)
    header = (f"# Shared {label} for {track}/{head_type} ({len(majority_slugs)} of {len(paths)} experiments — "
              f"auto-consolidated, identical after stripping header/noise). "
              f"Excluded (differ for real reasons): {', '.join(s for s in paths if s not in majority_slugs) or 'none'}\n\n")
    with open(shared_path, 'w', encoding='utf-8') as f:
        f.write(header + representative_content + "\n")
    for s in majority_slugs:
        os.remove(paths[s])
    return True


def get_active_model_key(staging, track, head_type, strategy, slug):
    """Read model_registry_key from that experiment's config.yaml — the
    actual MODEL_REGISTRY key (may repeat across configs on purpose; run_id
    is the unique-per-run identifier, this isn't)."""
    cfg_path = os.path.join(staging, "configs", track, head_type, f"{strategy}_{slug}.yaml")
    if not os.path.exists(cfg_path):
        return None
    for line in open(cfg_path, encoding='utf-8'):
        if line.startswith('model_registry_key:'):
            return line.split(':', 1)[1].strip()
    return None


def load_scores(staging):
    """Read staging/scores.csv (written by organize_by_type.sh) -> {slug: der}."""
    scores_path = os.path.join(staging, "scores.csv")
    scores = {}
    if not os.path.exists(scores_path):
        return scores
    with open(scores_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                scores[row["model_slug"]] = float(row["der"])
            except (KeyError, ValueError):
                pass
    return scores


def collapse_train(staging, track, head_type, rows):
    scores = load_scores(staging)
    slugs = [r["model_slug"] for r in rows]
    rows_by_slug = {r["model_slug"]: r for r in rows}
    train_dir = os.path.join(staging, "training", track, head_type)
    paths = {s: os.path.join(train_dir, f"{s}_finetune.py") for s in slugs}
    missing = {s: p for s, p in paths.items() if not os.path.exists(p)}
    paths = {s: p for s, p in paths.items() if os.path.exists(p)}
    if missing:
        print(f"  NOTE for {track}/{head_type}: {len(missing)} expected fine-tune file(s) not found on disk:")
        for s, p in missing.items():
            print(f"    missing: {p}")
    if len(paths) < 2:
        print(f"  SKIP finetune script collapse for {track}/{head_type}: only found {len(paths)} of {len(slugs)} expected files.")
        return False

    items = [(s, compare_key(open(p, encoding='utf-8').read(), mask_active_model=True)) for s, p in paths.items()]
    clusters = cluster_by_content(items)
    majority_content, majority_slugs = clusters[0]

    if len(clusters) > 1:
        outliers = [s for _, keys in clusters[1:] for s in keys]
        print(f"  finetune script for {track}/{head_type}: {len(majority_slugs)} of {len(paths)} are identical "
              f"(except ACTIVE_MODEL) and collapsed; {len(outliers)} genuinely differ and were LEFT SEPARATE: "
              f"{', '.join(outliers)} -- check these, they may reflect a real methodology change.")

    if len(majority_slugs) < 2:
        print(f"  SKIP finetune script collapse for {track}/{head_type}: no two files actually match -- all differ.")
        return False

    template_slug = majority_slugs[0]
    raw = strip_noise(HEADER_PATTERN.sub('', open(paths[template_slug], encoding='utf-8').read().replace('\r\n', '\n'), count=1))
    for s in majority_slugs:
        candidate = strip_noise(HEADER_PATTERN.sub('', open(paths[s], encoding='utf-8').read().replace('\r\n', '\n'), count=1))
        if 'cfg.export_include_weights:' in candidate:
            raw = candidate
            template_slug = s
            break

    best_row = min([rows_by_slug[s] for s in majority_slugs], key=lambda r: scores.get(r["model_slug"], float('inf')))
    best_key = get_active_model_key(staging, track, head_type, best_row["strategy"], best_row["model_slug"]) or "arabert_v02"
    best_der = scores.get(best_row["model_slug"])
    best_der_str = f"{best_der:.4f}" if best_der is not None else "n/a"

    def repl(m):
        return (
            'import argparse\n'
            '_parser = argparse.ArgumentParser()\n'
            f'_parser.add_argument("--active-model", type=str, default="{best_key}",\n'
            '                      choices=list(MODEL_REGISTRY.keys()),\n'
            f'                      help="Key into MODEL_REGISTRY (default: {best_key}, the best-scoring '
            f'run in this group at DER {best_der_str})")\n'
            '_args, _ = _parser.parse_known_args()\n'
            'ACTIVE_MODEL: str = _args.active_model'
        )
    raw, n = ACTIVE_MODEL_LINE.subn(repl, raw, count=1)
    if n != 1:
        print(f"  SKIP finetune script collapse for {track}/{head_type}: couldn't find ACTIVE_MODEL line to parametrize.")
        return False

    excluded = [s for s in paths if s not in majority_slugs]
    header = (f"# Shared fine-tuning script for {len(majority_slugs)} of {len(paths)} {track}/{head_type} "
              f"experiments (auto-consolidated -- identical except ACTIVE_MODEL; a broken "
              f"'!pip install kaggle' cell present in some copies was dropped).\n"
              f"# Excluded from this shared script (differ for real reasons, kept as their own file): "
              f"{', '.join(excluded) or 'none'}\n"
              f"# Defaults to --active-model {best_key}, the best-scoring run in this group "
              f"(DER {best_der_str}). Override with --active-model <key>.\n\n")
    shared_path = os.path.join(train_dir, f"finetune_{head_type}.py")
    with open(shared_path, 'w', encoding='utf-8') as f:
        f.write(header + raw.strip() + "\n")
    for s in majority_slugs:
        os.remove(paths[s])
    print(f"  Collapsed {len(majority_slugs)} fine-tune scripts -> {shared_path} (default: --active-model {best_key})")
    return True


def main(staging, map_file):
    groups = {}
    with open(map_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row["track"].strip(), row["head_type"].strip())
            groups.setdefault(key, []).append(row)

    for (track, head_type), rows in groups.items():
        slugs = [r["model_slug"].strip() for r in rows]
        if len(slugs) < 2:
            continue
        print(f"=== {track}/{head_type}: {len(slugs)} experiments ===")

        collapse_simple(
            staging, track, head_type, slugs,
            lambda st, t, h, s: os.path.join(st, "models", t, h, f"{s}_model.py"),
            os.path.join(staging, "models", track, head_type, f"{head_type}_model.py"),
            "model.py",
        )
        collapse_simple(
            staging, track, head_type, slugs,
            lambda st, t, h, s: os.path.join(st, "evaluation", t, h, f"{s}_evaluate.py"),
            os.path.join(staging, "evaluation", track, head_type, f"evaluate_{head_type}.py"),
            "evaluate.py",
        )
        collapse_train(staging, track, head_type, rows)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
