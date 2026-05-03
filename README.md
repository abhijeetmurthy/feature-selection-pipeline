# Feature Selection Pipeline

Enterprise-style ML pipeline for forward selection and backward elimination.

## Optimizations and Replacements
- Replaced custom 1-NN evaluation loop with `scikit-learn` leave-one-out validation (`KNeighborsClassifier` + `LeaveOneOut`).
- Added a NumPy fallback evaluator for environments without `scikit-learn`.
- Removed deprecated `pandas.DataFrame.append` usage and switched to efficient CSV streaming writes.
- Fixed backward-elimination implementation issues and standardized output naming.

## Structure
- `main.py`: CLI pipeline entrypoint.
- `forward_selection.py`: forward selection algorithm.
- `backward_elimination.py`: backward elimination algorithm.
- `helper.py`: optimized evaluation utilities.
- `scripts/run_pipeline.sh`: run wrapper.
- `requirements.txt`: Python dependencies.
- `configs/`, `docs/`: operational scaffolding.

## Quickstart
```bash
python3 -m pip install -r requirements.txt
./scripts/bootstrap.sh
./scripts/run_pipeline.sh --dataset large.txt --algorithm both
```

## CLI
```bash
python3 main.py --dataset <path> --algorithm <forward|backward|both>
```

## Example Datasets
- `small.txt`
- `large.txt`
- `1.txt`, `2.txt`, `3.txt`
