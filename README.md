# How to organize and push your experiments

This toolset takes a folder of raw Kaggle-notebook experiment exports
(notebook + config.json + dev_test_evaluation.json + confusion_matrix.png +
submission.csv, one folder per run) and turns them into a clean, reviewable
PR that matches the conventions in each folder's own `README.md`
(`configs/`, `models/`, `training/`, `evaluation/`, `experiments/`).

This does not make files runnable standalone —  it simply organizes them into the correct file structure.

Built and tested on Track 3 (Strategy A) — 12 experiments, two head
variants. Same scripts, same steps work for any track/strategy; you're
just filling in a different CSV.

## 0. One-time setup

- **Windows**: use **Git Bash**, not PowerShell or CMD. Right-click inside
  your folder → "Git Bash Here", or open Git Bash from the Start menu and
  `cd` into it. (`chmod`, the scripts, etc. don't work in PowerShell.)
- Python 3 with `pyyaml` installed: `pip install pyyaml --break-system-packages`
  (add `--break-system-packages` on newer Python — pip will complain
  otherwise, that's expected, not an error).
- Clone the repo once, normally: `git clone <repo-url>`.

## 1. Folder layout

Keep your raw experiment exports **outside** the actual git repo folder —
as a sibling, not inside it — so you can never accidentally `git add` the
whole unsorted dump:

```
your-workspace/
├── Raw Experiments/          <- your unzipped notebook exports, untouched
├── vocalization/              <- the actual git repo (has .gitignore, etc.)
├── organize_by_type.sh
├── consolidate.py
├── build_overviews.py
├── split_notebook.py
├── push_experiment.sh
├── push_all_experiments.sh
└── experiments_map.csv        <- YOUR filled-in copy of the template
```

All six scripts + your CSV need to sit **together**, next to (not inside)
the repo folder.

## 2. Fill in `experiments_map.csv`

Copy `experiments_map_TEMPLATE.csv` to `experiments_map.csv` and fill in
one row per experiment folder:

| Column | What it is | Example |
|---|---|---|
| `folder_name` | Exact name of your raw experiment folder (relative to wherever you point the script), including subfolders if nested | `Bilstm head/arabert_v02-0.9517` |
| `strategy` | Which of Strategy A/B/C/D (see `experiments/README.md`) — **not** the architecture | `strategy_a` |
| `model_slug` | A short unique id for this run (used in filenames) | `arabert_v02_09157` |
| `track` | Which track (Track 1 BiLSTM taggers / Track 2 char-level LLMs / Track 3 Arabic-pretrained transformers / Track 4 from-scratch transformer) | `track3` |
| `head_type` | The architecture variant within your track — whatever axis actually varies for you (e.g. `linear_head` vs `bilstm_crf_head`) | `bilstm_crf_head` |
| `public_score` / `private_score` | Kaggle leaderboard scores for that submission | `0.94743` / `0.95413` |

**`strategy` vs `track` vs `head_type` — these are three separate axes,
don't conflate them:**
- **Strategy** = what data you pretrained on before fine-tuning (A/B/C/D),
  defined in `experiments/README.md`.
- **Track** = which of the 4 broad architecture families you're in.
- **head_type** = whatever finer split matters *within* your track (for us,
  linear head vs BiLSTM-CRF head — yours might be something else entirely,
  or you might not need this axis at all, in which case just use one
  constant value like `default` for every row).

## 3. Run the pipeline

```bash
chmod +x organize_by_type.sh push_experiment.sh push_all_experiments.sh
./organize_by_type.sh "Raw Experiments" staging
```

Watch the terminal output. For each experiment it prints what it split the
notebook into; at the end, a `=== Consolidating ===` section tells you
whether `models/`, `evaluate.py`, and the fine-tuning script collapsed into
one shared file per group, or stayed separate. **Read this output** — see
§5 below for what it means.

Check `staging/` afterward:
```
staging/configs/<track>/<head_type>/<strategy>_<slug>.yaml       (one per run)
staging/models/<track>/<head_type>/<head_type>_model.py          (shared)
staging/training/<track>/<head_type>/finetune_<head_type>.py     (shared)
staging/evaluation/<track>/<head_type>/evaluate_<head_type>.py   (shared)
staging/evaluation/<track>/<head_type>/report_<strategy>_<slug>.md (one per run)
staging/experiments/<track>/<head_type>/<strategy>_overview.md   (one per group)
staging/leaderboard.md
```

## 4. Push

**Option A — one branch for everything** (simplest, what we used):
```bash
./push_all_experiments.sh vocalization staging <track>/<yourname>/All-experiments
```

**Option B — one branch per experiment** (more granular PRs):
```bash
./push_experiment.sh vocalization staging <track> <head_type> <slug>
```
repeat per row in your CSV.

Either way, open the PR on GitHub afterward (the site shows a banner right
after a push) — pushing never touches `main` directly.

## 5. Understanding the consolidation step

Your notebooks were probably built by copy-pasting one base notebook and
changing a config/model-name — that's normal, and `consolidate.py` handles
it automatically:

- It compares every `model.py` / `evaluate.py` / fine-tuning script within
  a `track/head_type` group.
- If they're identical (after stripping the auto-generated header and,
  for the training script, the active-model line), it collapses them into
  **one shared file**.
- If one genuinely differs — different code, not just a different model
  name — it's **left as its own separate file**, and the terminal tells
  you which one and why. **Don't ignore that message.** In our case it
  correctly caught a real bug fix (a self-training routine that was
  patched partway through the project) that would have been silently lost
  if we'd forced everything into one file.
- It also automatically strips known copy-paste noise: broken `!pip
  install ...` shell-magic cells (invalid outside a notebook) and
  reconciles blank-line-only differences.

If you get a `SKIP ... differ` message and you're not sure why, diff the
files yourself, or ask.

## 6. Known pitfalls (already hit these — don't repeat them)

- **`.gitignore` blocks `*.json`, `*.csv`, `*.txt` repo-wide.** That's why
  configs are `.yaml` and reports are `.md`, not `.json`/`.txt` — if you
  add a new file type, check it isn't silently ignored (`git status`
  should show it as untracked/new before you commit).
- **Confusion matrix images from the original notebooks are blank** — a
  `plt.show()`-before-`plt.savefig()` bug in the shared notebook. Not
  worth fixing retroactively without checkpoints; the per-class report in
  `evaluate.py`'s output covers the same information. Fix the order in
  your own notebook if you still have checkpoints to regenerate from.
- **Windows path with spaces**: always quote it, e.g.
  `./organize_by_type.sh "Raw Experiments" staging`.
- **`.md` reports look broken on GitHub if written as one big paragraph**
  — Markdown collapses single newlines. The script already formats reports
  properly (headings, bullet list, real table); if you're extending it,
  keep using actual Markdown syntax, not fixed-width text.
- **First push from a machine may need `git branch
  --set-upstream-to=origin/main main`** inside the repo folder if you get
  a "no tracking information" error on `git pull`.

## 7. Questions

Tag @YXlh-64 or whoever's reviewing PRs for your track before pushing if
you're unsure whether your `strategy`/`track`/`head_type` values are right
— getting the CSV wrong just means re-running the script, not real damage,
but getting the shared conventions right the first time keeps everyone's
PRs comparable.
