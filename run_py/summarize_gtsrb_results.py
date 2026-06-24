import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, stdev

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LEGACY_HISTORY_NAMES = {
    "wideresnet": "classical_legacy",
    "hybrid_qwideresnet": "hybrid_quantum",
}

MODEL_ALIASES = {
    "classical": "classical_legacy",
    "hybrid": "hybrid_quantum",
}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def seed_from_path(path, fallback=None):
    for part in os.path.abspath(path).split(os.sep):
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return fallback
    return fallback


def model_from_history_path(path, data):
    metadata = data.get("metadata", {})
    if metadata.get("model"):
        return metadata["model"]
    name = os.path.basename(path)
    if name.endswith("_gtsrb_history.json"):
        stem = name[: -len("_gtsrb_history.json")]
        return LEGACY_HISTORY_NAMES.get(stem, stem)
    return "unknown"


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def iter_files(root, suffix):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(dirpath, filename)


def collect_clean(results_dir):
    records = []
    for path in iter_files(results_dir, "_gtsrb_history.json"):
        data = read_json(path)
        metadata = data.get("metadata", {})
        seed = metadata.get("seed", seed_from_path(path))
        records.append({
            "seed": seed,
            "model": model_from_history_path(path, data),
            "best_val_acc": data.get("best_val_acc"),
            "best_epoch": data.get("best_epoch"),
            "test_acc": data.get("test_acc"),
            "history_path": path,
        })
    return records


def normalize_robustness_payload(data):
    if "results" in data:
        return data.get("metadata", {}), data["results"]
    return {}, data


def collect_robustness(results_dir):
    records = []
    for path in iter_files(results_dir, "gtsrb_robustness_results.json"):
        data = read_json(path)
        metadata, results = normalize_robustness_payload(data)
        seed = metadata.get("seed", seed_from_path(path))
        for model_name, model_results in results.items():
            model_name = normalize_model_name(model_name)
            for attack_name, values in model_results.items():
                epsilons = values.get("epsilon", [])
                accuracies = values.get("accuracy", [])
                asr_values = values.get("attack_success_rate", [])
                clean_acc = accuracies[0] if accuracies else None
                for idx, (epsilon, accuracy) in enumerate(zip(epsilons, accuracies)):
                    if idx < len(asr_values):
                        asr = asr_values[idx]
                    elif clean_acc:
                        asr = max(0.0, (clean_acc - accuracy) / clean_acc * 100.0)
                    else:
                        asr = None
                    records.append({
                        "seed": seed,
                        "model": model_name,
                        "attack": attack_name,
                        "epsilon": float(epsilon),
                        "accuracy": accuracy,
                        "attack_success_rate": asr,
                        "result_path": path,
                    })
    return records


def summarize_clean(records):
    grouped = defaultdict(list)
    for record in records:
        if record["test_acc"] is not None:
            grouped[record["model"]].append(record)

    rows = []
    for model_name, items in sorted(grouped.items()):
        seeds = [item["seed"] for item in items]
        val_values = [item["best_val_acc"] for item in items if item["best_val_acc"] is not None]
        test_values = [item["test_acc"] for item in items if item["test_acc"] is not None]
        val_mean, val_std = mean_std(val_values)
        test_mean, test_std = mean_std(test_values)
        rows.append({
            "model": model_name,
            "n": len(items),
            "seeds": ",".join(str(seed) for seed in seeds),
            "best_val_acc_mean": val_mean,
            "best_val_acc_std": val_std,
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
        })
    return rows


def summarize_robustness(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["model"], record["attack"], record["epsilon"])
        grouped[key].append(record)

    rows = []
    for (model_name, attack_name, epsilon), items in sorted(grouped.items()):
        seeds = [item["seed"] for item in items]
        acc_values = [item["accuracy"] for item in items if item["accuracy"] is not None]
        asr_values = [item["attack_success_rate"] for item in items if item["attack_success_rate"] is not None]
        acc_mean, acc_std = mean_std(acc_values)
        asr_mean, asr_std = mean_std(asr_values)
        rows.append({
            "model": model_name,
            "attack": attack_name,
            "epsilon": epsilon,
            "n": len(items),
            "seeds": ",".join(str(seed) for seed in seeds),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "attack_success_rate_mean": asr_mean,
            "attack_success_rate_std": asr_std,
        })
    return rows


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def write_text_summary(path, clean_rows, robustness_rows):
    with open(path, "w") as f:
        f.write("GTSRB MULTI-SEED SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Clean accuracy\n")
        f.write("-" * 80 + "\n")
        for row in clean_rows:
            f.write(
                f"{row['model']:<24} n={row['n']:<2} "
                f"test={format_float(row['test_acc_mean'])}±{format_float(row['test_acc_std'])} "
                f"val={format_float(row['best_val_acc_mean'])}±{format_float(row['best_val_acc_std'])}\n"
            )

        f.write("\nRobust accuracy\n")
        f.write("-" * 80 + "\n")
        for row in robustness_rows:
            f.write(
                f"{row['model']:<24} {row['attack']:<5} eps={row['epsilon']:<10.6f} "
                f"acc={format_float(row['accuracy_mean'])}±{format_float(row['accuracy_std'])} "
                f"asr={format_float(row['attack_success_rate_mean'])}±{format_float(row['attack_success_rate_std'])}\n"
            )


def plot_robustness(rows, path):
    if plt is None:
        print("matplotlib is not installed; skipping summary_robustness.png")
        return
    attacks = sorted({row["attack"] for row in rows})
    models = sorted({row["model"] for row in rows})
    if not attacks or not models:
        return

    fig, axes = plt.subplots(1, len(attacks), figsize=(6 * len(attacks), 5), squeeze=False)
    axes = axes[0]
    markers = ["o", "s", "^", "D", "x"]
    for ax, attack in zip(axes, attacks):
        for model_idx, model_name in enumerate(models):
            subset = [row for row in rows if row["attack"] == attack and row["model"] == model_name]
            subset = sorted(subset, key=lambda row: row["epsilon"])
            eps = [row["epsilon"] for row in subset]
            acc = [row["accuracy_mean"] / 100.0 for row in subset]
            std = [row["accuracy_std"] / 100.0 for row in subset]
            ax.plot(eps, acc, marker=markers[model_idx % len(markers)], linewidth=2, label=model_name)
            ax.fill_between(eps, np_sub(acc, std), np_add(acc, std), alpha=0.12)
        ax.set_title(attack.upper())
        ax.set_xlabel("Perturbation budget")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.01, 1.01)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def np_sub(left, right):
    return [max(0.0, a - b) for a, b in zip(left, right)]


def np_add(left, right):
    return [min(1.0, a + b) for a, b in zip(left, right)]


def ablation_rows(clean_rows, robustness_rows):
    rows = []
    clean_by_model = {row["model"]: row for row in clean_rows}
    for model_name in sorted(clean_by_model):
        if model_name.startswith("hybrid") or model_name == "classical_strong":
            clean = clean_by_model[model_name]
            rows.append({
                "model": model_name,
                "metric": "clean_test_acc",
                "attack": "",
                "epsilon": "",
                "mean": clean["test_acc_mean"],
                "std": clean["test_acc_std"],
                "n": clean["n"],
            })
    for row in robustness_rows:
        if row["model"].startswith("hybrid") or row["model"] == "classical_strong":
            rows.append({
                "model": row["model"],
                "metric": "robust_accuracy",
                "attack": row["attack"],
                "epsilon": row["epsilon"],
                "mean": row["accuracy_mean"],
                "std": row["accuracy_std"],
                "n": row["n"],
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multi-seed GTSRB experiment outputs.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    clean_records = collect_clean(args.results_dir)
    robustness_records = collect_robustness(args.results_dir)
    clean_rows = summarize_clean(clean_records)
    robustness_rows = summarize_robustness(robustness_records)
    ablation = ablation_rows(clean_rows, robustness_rows)

    clean_fields = [
        "model", "n", "seeds", "best_val_acc_mean", "best_val_acc_std", "test_acc_mean", "test_acc_std",
    ]
    robustness_fields = [
        "model", "attack", "epsilon", "n", "seeds",
        "accuracy_mean", "accuracy_std", "attack_success_rate_mean", "attack_success_rate_std",
    ]
    ablation_fields = ["model", "metric", "attack", "epsilon", "mean", "std", "n"]

    write_csv(os.path.join(output_dir, "summary_clean.csv"), clean_rows, clean_fields)
    write_json(os.path.join(output_dir, "summary_clean.json"), clean_rows)
    write_csv(os.path.join(output_dir, "summary_robustness.csv"), robustness_rows, robustness_fields)
    write_json(os.path.join(output_dir, "summary_robustness.json"), robustness_rows)
    write_csv(os.path.join(output_dir, "summary_ablation.csv"), ablation, ablation_fields)
    write_json(os.path.join(output_dir, "summary_ablation.json"), ablation)
    write_text_summary(os.path.join(output_dir, "summary_robustness.txt"), clean_rows, robustness_rows)
    plot_robustness(robustness_rows, os.path.join(output_dir, "summary_robustness.png"))

    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
