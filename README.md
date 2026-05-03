# Feature Selection Pipeline

Generic machine-learning feature-selection pipeline with forward selection and backward elimination.

## Pipeline Stages

1. Ingest: read dataset matrix from text files.
2. Evaluate: score feature subsets with nearest-neighbor baseline.
3. Select: run forward selection and backward elimination.
4. Track: export intermediate rankings and accuracy logs.
5. Visualize: generate plots for selected feature trajectories.

## Repository Layout

- `main.py`: pipeline driver.
- `helper.py`: evaluation utilities.
- `forward_selection.py`: forward-search selector.
- `backward_elimination.py`: backward-search selector.
- `my_plots.py`: plotting helpers.

## Example Datasets (Current)

- `small.txt`
- `large.txt`
- `1.txt`, `2.txt`, `3.txt`

## Example Outputs (Current)

- `*_forward.csv`, `*_backward.csv`
- `output_forward_*.txt`, `output_backward_*.txt`
- `*_forward.png`, `*_backward.png`

## Run

```bash
python main.py
```
