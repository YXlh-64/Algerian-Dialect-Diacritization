"""Split a notebook into models/ + training/ + evaluation/ files by section header."""
import json, sys, re

def split_notebook(nb_path, slug, track, out_dir):
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = nb['cells']

    sections = []  # list of (header, [code_lines])
    preamble = []
    current_header = None
    current_lines = []

    def flush():
        if current_header is not None:
            sections.append((current_header, current_lines[:]))

    for cell in cells:
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown' and re.match(r'^##\s', src):
            flush()
            current_header = src.split('\n')[0].strip('# ').strip()
            current_lines = []
        elif cell['cell_type'] == 'code':
            if current_header is None:
                preamble.append(src)
            else:
                current_lines.append(src)
    flush()

    def route(header):
        h = header.lower()
        # Word-boundary match, not substring — "8. Train `ACTIVE_MODEL`" must
        # NOT match "model" here (it contains "active_model" as one token,
        # no boundary before "model"), or the whole training loop gets
        # misrouted into models/*.py instead of training/*.py.
        if re.search(r'\bmodel\b', h) and not re.search(r'\bevaluat', h):
            return 'models'
        if re.search(r'\bevaluat', h):
            return 'evaluation'
        return 'training'

    buckets = {'models': [], 'training': [], 'evaluation': []}
    for header, lines in sections:
        buckets[route(header)].append((header, lines))

    written = []

    # Extract import statements from anywhere in the notebook (imports
    # usually live in a "## 1. Environment" style section, not literally
    # before the first header) so every output file has its basic
    # type/library imports resolved, not just the training file.
    import_lines = []
    seen_imports = set()
    all_code_blocks = list(preamble) + [line for _, lns in sections for line in lns]
    for block in all_code_blocks:
        for line in block.split('\n'):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and stripped not in seen_imports:
                import_lines.append(stripped)  # dedented — these become top-level statements
                seen_imports.add(stripped)

    for bucket, items in buckets.items():
        if not items:
            continue
        suffix = {'models': 'model', 'training': 'train', 'evaluation': 'evaluate'}[bucket]
        out_path = f"{out_dir}/{bucket}/{track}/{slug}_{suffix}.py"
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-split from {nb_path} for {slug} ({track})\n")
            f.write(f"# NOTE: shared imports/setup live in training/{track}/{slug}_train.py\n")
            f.write("# Sanity-check imports here before relying on this file standalone.\n\n")
            if bucket != 'training' and import_lines:
                f.write("# --- Imports (copied from the notebook's preamble so this file has its\n")
                f.write("#     basic dependencies resolved; full setup still lives in training/) ---\n")
                f.write('\n'.join(import_lines))
                f.write('\n\n')
            if bucket == 'training':
                f.write("# --- Environment & setup (preamble cells before first ## section) ---\n")
                f.write('\n\n'.join(preamble))
                f.write('\n\n')
            for header, lines in items:
                f.write(f"# ## {header}\n")
                f.write('\n\n'.join(lines))
                f.write('\n\n')
        written.append(out_path)
    return written

if __name__ == '__main__':
    nb_path, slug, track, out_dir = sys.argv[1:5]
    for p in split_notebook(nb_path, slug, track, out_dir):
        print("wrote", p)
