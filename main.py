import argparse
from pathlib import Path

import numpy as np

from backward_elimination import backward_elimination
from forward_selection import forward_selection
from helper import evaluate_subset


def parse_args():
    parser = argparse.ArgumentParser(description="Feature selection pipeline")
    parser.add_argument(
        "--dataset",
        default="large.txt",
        help="Path to dataset file (default: large.txt)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["forward", "backward", "both"],
        default="both",
        help="Selection algorithm to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.genfromtxt(dataset_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Dataset must be a 2D matrix with label column + features")

    n_features = data.shape[1] - 1
    full_accuracy = evaluate_subset(data, list(range(1, n_features + 1)))

    print(f"Dataset: {dataset_path}")
    print(f"Instances: {data.shape[0]} | Features: {n_features}")
    print(f"1-NN leave-one-out accuracy with all features: {full_accuracy:.4f}%")

    run_name = dataset_path.stem

    if args.algorithm in ("forward", "both"):
        forward_selection(data, run_name)

    if args.algorithm in ("backward", "both"):
        backward_elimination(data, run_name)


if __name__ == "__main__":
    main()
