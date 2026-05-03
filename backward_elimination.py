import csv

from helper import evaluate_subset


def backward_elimination(data, run_name):
    """Greedy backward elimination with leave-one-out 1-NN scoring."""
    n_features_total = data.shape[1] - 1
    current = list(range(1, n_features_total + 1))
    best_global_features = current.copy()
    best_global_accuracy = evaluate_subset(data, current)
    rows = []

    for _ in range(n_features_total - 1):
        best_local_drop = None
        best_local_accuracy = -1.0

        for feature_idx in current:
            candidate = [f for f in current if f != feature_idx]
            if not candidate:
                continue

            accuracy = evaluate_subset(data, candidate)
            rows.append({"features": candidate.copy(), "accuracy": round(accuracy, 6)})
            print(f"Features considered: {candidate} -> accuracy {accuracy:.4f}%")

            if accuracy > best_local_accuracy:
                best_local_accuracy = accuracy
                best_local_drop = feature_idx

        if best_local_drop is None:
            break

        current = [f for f in current if f != best_local_drop]
        print(f"Remaining feature set this round: {current} ({best_local_accuracy:.4f}%)")

        if best_local_accuracy > best_global_accuracy:
            best_global_accuracy = best_local_accuracy
            best_global_features = current.copy()

    output_path = f"{run_name}_backward.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["features", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Backward elimination best features: {best_global_features} "
        f"with accuracy {best_global_accuracy:.4f}%"
    )
    return best_global_features, best_global_accuracy
