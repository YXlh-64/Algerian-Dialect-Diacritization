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

## Important Notes
- No executable tests are included in this initialization stage.
- Prioritize deterministic tests and fixed fixtures when implementations are added.
