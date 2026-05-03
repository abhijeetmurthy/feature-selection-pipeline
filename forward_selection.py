import csv

from helper import evaluate_subset


def forward_selection(data, run_name):
    """Greedy forward feature selection with leave-one-out 1-NN scoring."""
    n_features_total = data.shape[1] - 1
    selected = []
    best_global_features = []
    best_global_accuracy = -1.0
    rows = []

    for _ in range(n_features_total):
        best_local_feature = None
        best_local_accuracy = -1.0

        for feature_idx in range(1, n_features_total + 1):
            if feature_idx in selected:
                continue

            candidate = selected + [feature_idx]
            accuracy = evaluate_subset(data, candidate)
            rows.append({"features": candidate.copy(), "accuracy": round(accuracy, 6)})
            print(f"Features considered: {candidate} -> accuracy {accuracy:.4f}%")

            if accuracy > best_local_accuracy:
                best_local_accuracy = accuracy
                best_local_feature = feature_idx

        if best_local_feature is None:
            break

        selected.append(best_local_feature)
        print(f"Selected feature set this round: {selected} ({best_local_accuracy:.4f}%)")

        if best_local_accuracy > best_global_accuracy:
            best_global_accuracy = best_local_accuracy
            best_global_features = selected.copy()

    output_path = f"{run_name}_forward.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["features", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Forward selection best features: {best_global_features} "
        f"with accuracy {best_global_accuracy:.4f}%"
    )
    return best_global_features, best_global_accuracy
