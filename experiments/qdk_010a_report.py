import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "qdk_009" / "QDK-009-STATISTICAL-VALIDATION.json"
OUT = ROOT / "results" / "qdk_010a"
REPORT = ROOT / "reports" / "qdk_010a"

OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)


def load_results():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    with open(INPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data):
    between = data.get("between_structure_distance", {})
    within = data.get("within_structure_variation", {})

    return between, within


def safe_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    return None


def make_summary(data):
    between, within = extract_metrics(data)

    distances = {
        k: safe_number(v)
        for k, v in between.items()
    }

    within_mean = {
        k: safe_number(v.get("mean"))
        for k, v in within.items()
        if isinstance(v, dict)
    }

    within_std = {
        k: safe_number(v.get("std"))
        for k, v in within.items()
        if isinstance(v, dict)
    }

    min_between = min(
        v for v in distances.values()
        if v is not None
    )

    max_within_mean = max(
        v for v in within_mean.values()
        if v is not None
    )

    separation_ratio = (
        min_between / max_within_mean
        if max_within_mean > 0
        else math.inf
    )

    return {
        "experiment": "QDK-010A",
        "source_experiment": "QDK-009",
        "between_structure_distance": distances,
        "within_structure_mean": within_mean,
        "within_structure_std": within_std,
        "minimum_between_distance": min_between,
        "maximum_within_mean_variation": max_within_mean,
        "separation_ratio": separation_ratio,
    }


def save_json(summary):
    path = OUT / "QDK-010A-SUMMARY.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved: {path}")


def make_distance_plot(summary):
    distances = summary["between_structure_distance"]

    labels = list(distances.keys())
    values = list(distances.values())

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("Trajectory distance")
    plt.title("QDK-010A Between-Structure Separation")
    plt.xticks(rotation=20)
    plt.tight_layout()

    path = OUT / "QDK-010A-between-distance.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")


def make_within_plot(summary):
    means = summary["within_structure_mean"]
    stds = summary["within_structure_std"]

    labels = list(means.keys())
    values = [means[x] for x in labels]
    errors = [stds.get(x, 0.0) for x in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, yerr=errors, capsize=5)
    plt.ylabel("Within-structure variation")
    plt.title("QDK-010A Within-Structure Variation")
    plt.xticks(rotation=20)
    plt.tight_layout()

    path = OUT / "QDK-010A-within-variation.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")


def latex_escape(text):
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def make_latex(summary):
    ratio = summary["separation_ratio"]

    rows = []

    for name, value in summary["between_structure_distance"].items():
        rows.append(
            f"{latex_escape(name)} & {value:.6f} \\\\"
        )

    table = "\n".join(rows)

    latex = rf"""
\documentclass[11pt]{{article}}

\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}

\geometry{{margin=1in}}

\title{{QDK-010A: Noise-Response Statistical Reporting}}
\author{{Tai D. Nguyen}}
\date{{\today}}

\begin{{document}}

\maketitle

\begin{{abstract}}
This report analyzes the QDK noise-response trajectories generated
under the reference stochastic model. The objective is to quantify
whether different encoded information structures produce separable
noise-response signatures.
\end{{abstract}}

\section{{Objective}}

QDK-010A evaluates the statistical separation of the three reference
structures:

\begin{{itemize}}
\item STRUCTURED
\item WEAK
\item ASYMMETRIC
\end{{itemize}}

The experiment is a numerical validation stage and does not constitute
validation on physical quantum hardware.

\section{{Between-Structure Separation}}

\begin{{table}}[h]
\centering
\begin{{tabular}}{{lr}}
\toprule
Comparison & Distance \\
\midrule
{table}
\bottomrule
\end{{tabular}}
\caption{{Pairwise trajectory distances.}}
\end{{table}}

\section{{Separation Ratio}}

The observed minimum between-structure distance is

\[
D_{{\min}} = {summary["minimum_between_distance"]:.6f}.
\]

The maximum mean within-structure variation is

\[
V_{{\max}} = {summary["maximum_within_mean_variation"]:.6f}.
\]

The resulting descriptive separation ratio is

\[
R = \frac{{D_{{\min}}}}{{V_{{\max}}}}
  = {ratio:.6f}.
\]

This ratio is a descriptive metric. It is not a p-value, confidence
interval, or standardized statistical effect size.

\section{{Interpretation}}

The numerical experiment shows that the three encoded structures
produce distinguishable trajectories under the specified reference
noise model.

The result supports the hypothesis that the simulated noise response
can retain information about the encoded structure.

However, this result alone does not establish that physical quantum
noise universally acts as an information channel. Hardware-level
validation remains necessary.

\section{{Reproducibility}}

The analysis is derived from the QDK-009 result artifact:

\begin{{verbatim}}
results/qdk_009/QDK-009-STATISTICAL-VALIDATION.json
\end{{verbatim}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.9\textwidth]{{../../results/qdk_010a/QDK-010A-between-distance.png}}
\caption{{Between-structure trajectory separation.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.85\textwidth]{{../../results/qdk_010a/QDK-010A-within-variation.png}}
\caption{{Within-structure stochastic variation.}}
\end{{figure}}

\section{{Conclusion}}

QDK-010A provides an automated statistical reporting layer for the
QDK noise-as-information research program.

The evidence is currently computational and model-dependent.
The next validation stage is QDK-010 hardware execution followed by
the same analysis pipeline.

\end{{document}}
"""

    path = REPORT / "QDK-010A-report.tex"

    with open(path, "w", encoding="utf-8") as f:
        f.write(latex.strip() + "\n")

    print(f"Saved: {path}")


def main():
    print("=" * 60)
    print("QDK-010A — STATISTICAL REPORTING PIPELINE")
    print("=" * 60)

    data = load_results()
    summary = make_summary(data)

    save_json(summary)
    make_distance_plot(summary)
    make_within_plot(summary)
    make_latex(summary)

    print("=" * 60)
    print("QDK-010A COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
