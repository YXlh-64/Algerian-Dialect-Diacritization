# experiments/

## Purpose
This folder will organize experiment plans, strategy-level run definitions, and entry points.

## Expected Contents
The four planned transfer strategies are:
- Strategy A: No pretraining, fine-tune/train on Algerian only.
- Strategy B: Pretrain on MSA (Tashkeela ~50k), then fine-tune on Algerian.
- Strategy C: Pretrain on NADI (~15k), then fine-tune on Algerian.
- Strategy D: Pretrain on MSA + NADI (~65k), then fine-tune on Algerian.

## Naming Convention
Use strategy-prefixed naming once files are introduced, for example:
- strategy_a_*
- strategy_b_*
- strategy_c_*
- strategy_d_*

## Important Notes
- Entry point scripts are intentionally not added yet.
- Ensure each strategy maps cleanly to reproducible config selections.
