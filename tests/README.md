# tests/

## Purpose
This folder will contain the test suite ensuring correctness and reproducibility of the project.

## Expected Contents
Planned testing coverage includes:
- Unit tests for preprocessing, tokenization, and metric functions.
- Integration tests for training/evaluation pipeline wiring.
- Regression tests for core experiment behavior.

## Naming Convention
When tests are added, use clear scope-based names, for example:
- test_utils_*
- test_models_*
- test_training_*
- test_evaluation_*

## Running

Track-1 utility and focal-loss regression tests use the standard-library test runner:

```bash
python -m unittest -v tests/test_track1_utils.py
```

PyTorch-dependent cases are skipped automatically when PyTorch is unavailable.

## Important Notes
- Keep tests deterministic and use fixed fixtures.
- Add integration coverage for long-running training separately from lightweight unit tests.
