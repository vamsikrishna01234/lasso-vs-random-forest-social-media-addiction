"""Compare Lasso Regression with Random Forest for social media addiction detection.

For each of N iterations (default 10) the data is split 70/30 with a new random
seed, every model is trained and tested on that split, and accuracy, macro
precision / recall / F1 and prediction time are recorded. An Independent Samples
T-Test (with Levene's test for equal variances) then compares each Lasso model
with Random Forest.

Usage (run from the project folder):
    python src/run_experiment.py
    python src/run_experiment.py --iterations 10 --features all
    python src/run_experiment.py --features report3
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_utils import ROOT, build_features, load_dataset, split_and_scale
from models import (COMPARISON_MODEL, LassoLogisticClassifier,
                    LassoRegressionClassifier)

MODEL_CLASSES = [LassoRegressionClassifier, LassoLogisticClassifier, COMPARISON_MODEL]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def run_iterations(X, y, iterations, base_seed):
    feature_names = list(X.columns)
    rows = []
    for i in range(iterations):
        seed = base_seed + i
        X_tr, X_te, y_tr, y_te = split_and_scale(X, y, seed)
        line = [f"Iteration {i + 1:>2}/{iterations}"]
        for cls in MODEL_CLASSES:
            model = cls(random_state=seed).fit(X_tr, y_tr)

            start = time.perf_counter()
            pred = model.predict(X_te)
            predict_seconds = time.perf_counter() - start

            acc = accuracy_score(y_te, pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, pred, average="macro", zero_division=0
            )
            rows.append({
                "iteration": i + 1, "model": cls.name, "accuracy": acc * 100,
                "precision_macro": prec * 100, "recall_macro": rec * 100,
                "f1_macro": f1 * 100,
                "predict_us_per_record": predict_seconds / len(X_te) * 1e6,
            })
            line.append(f"{cls.name} {acc * 100:6.2f}%")

            if i == 0 and isinstance(model, LassoRegressionClassifier):
                kept = model.selected_features(feature_names)
                print(f"Lasso kept {len(kept)} of {len(feature_names)} features: {kept}")
        print(" | ".join(line))
    return pd.DataFrame(rows)


def summarise(results):
    grouped = results.groupby("model", sort=False)
    summary = grouped.agg(
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        sem_accuracy=("accuracy", "sem"),
        mean_precision_macro=("precision_macro", "mean"),
        mean_recall_macro=("recall_macro", "mean"),
        mean_f1_macro=("f1_macro", "mean"),
        predict_us_per_record=("predict_us_per_record", "mean"),
    )
    return summary.round(4)


def statistical_tests(results):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    other = COMPARISON_MODEL.name
    b = results.loc[results["model"] == other, "accuracy"].to_numpy()
    rows = []
    for cls in (LassoRegressionClassifier, LassoLogisticClassifier):
        a = results.loc[results["model"] == cls.name, "accuracy"].to_numpy()
        levene = stats.levene(a, b, center="mean")  # same centre SPSS uses
        t_eq = stats.ttest_ind(a, b, equal_var=True)
        t_welch = stats.ttest_ind(a, b, equal_var=False)
        rows.append({
            "comparison": f"{cls.name} vs {other}",
            "mean_difference": round(a.mean() - b.mean(), 4),
            "levene_F": round(float(levene.statistic), 4),
            "levene_p": f"{levene.pvalue:.3g}",
            "t_equal_var": round(float(t_eq.statistic), 4),
            "p_equal_var": f"{t_eq.pvalue:.3g}",
            "t_welch": round(float(t_welch.statistic), 4),
            "p_welch": f"{t_welch.pvalue:.3g}",
        })
    return pd.DataFrame(rows)


def plot_accuracy(summary, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(summary.index)
    means = summary["mean_accuracy"].to_numpy()
    errors = 2 * summary["sd_accuracy"].fillna(0).to_numpy()
    ax.bar(names, means, yerr=errors, capsize=8, color="#5B9BD5", edgecolor="black", width=0.55)
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Mean Accuracy by Model")
    ax.set_ylim(0, 110)
    ax.text(0.5, -0.17, "Error bars: +/- 2 SD", ha="center", transform=ax.transAxes, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=10, help="runs per model (default 10)")
    parser.add_argument("--features", choices=["all", "report3"], default="all",
                        help="feature set (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="first random seed (default 0)")
    args = parser.parse_args()

    df = load_dataset()
    X, y = build_features(df, args.features)
    print(f"Features used ({X.shape[1]}): {list(X.columns)}")
    print(f"Majority-class baseline (always predict 'High'): {(y == 2).mean() * 100:.2f}%\n")

    results = run_iterations(X, y, args.iterations, args.seed)
    summary = summarise(results)
    tests = statistical_tests(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "accuracy_per_iteration.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv")
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    plot_accuracy(summary, PLOTS_DIR / "experiment_accuracy.png")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Summary over", args.iterations, "iterations (values in %) ===")
    print(summary.to_string())
    print("\n=== Independent Samples T-Test on accuracy ===")
    print(tests.to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR} and the chart to {PLOTS_DIR / 'experiment_accuracy.png'}")


if __name__ == "__main__":
    main()
