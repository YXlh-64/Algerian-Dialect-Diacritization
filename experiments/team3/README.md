# Team 3: §6 baselines and §7 LLM comparison

Code that produces the numbers and tables of the paper's §6 (Baseline Models) and §7 (LLM Comparison). **Corpus-derived outputs never go in this public repo.** Prompts with sentences, LLM outputs, per-sentence files and paper drafts live in a local `team3_deliverables/` folder that is shared on Drive. Pass its path with `--out-dir` / `--llm-dir`.

## LLM benchmark (`llm_benchmark/`)

1. `build_batches.py` freezes the protocol. It writes `{sent_id, input}` files (gold targets are never read) and self-contained batch prompts for the `zero_shot` and `five_shot` conditions (5 training demonstrations, one per length quintile, seed 42). It also writes a manifest with prompt hashes.
2. Each batch prompt is answered by the LLM in a fresh context, with lines in the form `ID<TAB>vocalized sentence`. For the paper run, Claude Opus 5.5 (`claude-opus-5-5`) was queried through Claude Code on 15 Oct 2026, one fresh agent per batch, with access only to its own prompt file and output file.
3. `collect_outputs.py` merges the batches and checks that every sentence is present. For the test split it also writes skeleton-repaired lines for `make_submission.py`.
4. Score with `python -m evaluation.common.score` (dev).
5. `analyze.py` runs the OOV / ambiguous-word buckets, dialect position rates (word-initial sukun, word-final vowels/tanwin), top confusions, sentence-level paired bootstrap, and example mining.

```bash
python experiments/team3/llm_benchmark/build_batches.py --data-root data --out-dir $D/llm --split dev --condition five_shot
python -m experiments.team3.llm_benchmark.collect_outputs --llm-dir $D/llm --condition five_shot --split dev
python -m evaluation.common.score --gold data/dev_data/dev_Algerian-DIAC.jsonl \
  --pred $D/llm/claude-opus-5-5/predictions/five_shot_dev.jsonl --out $D/llm/claude-opus-5-5/scores/five_shot_dev \
  --model-name "claude-opus-5-5 (five_shot)" --track llm --split dev --system-type llm \
  --append-master $D/results/MASTER_RESULTS_scored.csv
python -m experiments.team3.llm_benchmark.analyze --gold data/dev_data/dev_Algerian-DIAC.jsonl \
  --train data/train_data/train_Algerian-DIAC.jsonl \
  --system llm_zero=... --system llm_five=... --system t4_neural=... --system t4_v2=... \
  --compare llm_five:t4_neural --compare llm_five:t4_v2 --compare llm_five:llm_zero \
  --examples-for llm_five --reference-system t4_v2 --out $D/llm/analysis
```

## Results and tables (`results/`)

- `reported_results.py` is the registry of the numbers each track reported, with source files and scope notes (full dev / Track 3's 20% dev slice / unknown). The Kaggle public/private micro-F1 is the cross-track comparison.
- `build_master_table.py` (expects the four analysis systems `llm_zero`, `llm_five`, `t4_neural`, `t4_v2`; missing ones are skipped in `tab_llm.tex`) merges the registry with the shared-scorer rows into `MASTER_RESULTS.csv` and generates `tab_main_results.tex`, `tab_llm.tex` and `MAIN_RESULTS.md`. Tables are never edited by hand. Pass `--llm-kaggle PUBLIC PRIVATE` once the LLM submission is scored, and `--style cas` for the Elsevier CAS manuscript (`tabular*` with `\tblwidth`, `tbl:` labels) instead of the plain article layout.
- `audit_numbers.py` lists any decimal in the paper prose that cannot be traced to the results files.
