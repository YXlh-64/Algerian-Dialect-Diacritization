# configs/

## Purpose
This folder will contain experiment configuration files that define model, data, optimization, and runtime settings.

## Expected Contents
- One configuration file per experiment strategy and variant.
- Configuration templates and validated final configs for reproducible runs.

## Naming Convention
Use clear, strategy-centered names once configs are added, for example:
- strategy_a_baseline
- strategy_b_msa_transfer
- strategy_c_nadi_transfer
- strategy_d_combined_transfer

## Important Notes
- Configuration files are not added yet in this initialization stage.
- Keep configurations deterministic and fully documented to support reproducibility.
