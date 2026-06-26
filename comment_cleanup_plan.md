# Comment Cleanup Plan

## Goal

Reduce excessive commenting and obvious AI-assisted phrasing across the codebase while preserving short, useful docstrings and the few comments that explain non-obvious behavior.

## Reference Style

Use `src/blackjack_predictor/api_specialized.py` as the style reference.

Keep:

- Short public docstrings
- Comments that explain non-obvious constraints or behavior

Remove:

- Numbered section comments
- Banner comments
- Conversational or tutorial-style comments
- Comments that restate the next line of code
- AI-signature phrasing such as overexplained process narration

## Scope

### Primary package files

1. `src/blackjack_predictor/api.py`
2. `src/blackjack_predictor/data_drift.py`
3. `src/blackjack_predictor/data_/preprocessing.py`
4. `src/blackjack_predictor/data_/datamodule.py`
5. `src/blackjack_predictor/improve_speed.py`

### Secondary package files

1. `src/blackjack_predictor/data_/dataset.py`
2. `src/blackjack_predictor/train.py`
3. `src/blackjack_predictor/models/ffnn.py`

### Test files

1. `tests/test_training_procedure.py`
2. `tests/test_preprocessing.py`
3. `tests/test_data.py`

### Files that appear already consistent

1. `src/blackjack_predictor/api_specialized.py`
2. `src/blackjack_predictor/export_onnx.py`
3. `src/blackjack_predictor/evaluate.py`
4. `src/blackjack_predictor/logger.py`
5. `src/blackjack_predictor/tasks.py`
6. `src/blackjack_predictor/run_sweep.py`
7. `src/blackjack_predictor/profiling.py`
8. `src/blackjack_predictor/optimize_and_drift_test.py`
9. `src/blackjack_predictor/data_/dataset_statistics.py`
10. `src/blackjack_predictor/visualize.py`
11. `tests/test_api.py`
12. `tests/test_onnx_alignment.py`
13. `tests/locustfile.py`

## Execution Order

1. Clean `src/blackjack_predictor/api.py` first and use it as the baseline for the rest of the cleanup.
2. Clean the remaining primary package files.
3. Clean the secondary package files.
4. Clean the flagged test files.
5. Leave already-consistent files unchanged unless a clearly unnecessary comment is found during the pass.

## File-by-File Intent

### `src/blackjack_predictor/api.py`

- Remove numbered section comments and milestone banners
- Simplify overly explanatory docstrings
- Remove comments that narrate obvious tensor creation, logging, and summary-building steps
- Keep only comments that explain non-obvious runtime constraints if needed

### `src/blackjack_predictor/data_drift.py`

- Remove step-by-step numbered comments
- Replace with a short function docstring if needed

### `src/blackjack_predictor/data_/preprocessing.py`

- Remove obvious inline comments for loading, filtering, tensor conversion, and saving
- Tighten the function docstring

### `src/blackjack_predictor/data_/datamodule.py`

- Remove banner comments
- Remove unnecessary initialization comment

### `src/blackjack_predictor/improve_speed.py`

- Reduce the long explanatory note in the docstring to a short, factual description

### `src/blackjack_predictor/data_/dataset.py`

- Remove the emphatic explanatory comment about feature slicing

### `src/blackjack_predictor/train.py`

- Remove the overexplained GPU strategy comment
- Remove stray `# test`

### `src/blackjack_predictor/models/ffnn.py`

- Remove the demo comment in `__main__`

### `tests/test_training_procedure.py`

- Remove banner comments and numbered step comments
- Shorten verbose docstrings
- Keep only minimal explanation for non-obvious test helpers

### `tests/test_preprocessing.py`

- Remove conversational comments and repeated assertion narration
- Keep at most one short note if the test data setup needs clarification

### `tests/test_data.py`

- Remove numbered and tutorial-style comments
- Let test names and assertions carry the intent

## Editing Rules

1. Prefer deletion over rewriting.
2. Make the smallest possible changes.
3. Avoid behavioral changes.
4. Preserve docstrings for public functions and classes, but keep them short.
5. Add comments only when the code would otherwise be hard to understand.

## Verification

1. Run targeted tests for touched files.
2. Run the broader test suite if needed after the cleanup pass.
3. Run formatting or linting only if the edits require it.
