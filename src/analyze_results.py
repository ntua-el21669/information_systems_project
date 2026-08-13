"""Create statistical summaries and charts from evaluation CSVs.

The script uses the same metric used in the project:
an exact execution match is correct only when it is not a
trivial empty-result match.  GPT and Qwen are compared with an exact paired
McNemar test because every model answered the same sampled questions.

Usage:
    python src/analyze_results.py

Outputs are written to data/results/analysis/.
"""

from __future__ import annotations

import csv
import html
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


# Resolve project files relative to this script, not the caller's current
# directory. This supports both `python src/analyze_results.py` from the
# repository root and `python analyze_results.py` from `src/`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis"
MODEL_FILES = {
    "GPT-4o-mini": RESULTS_DIR / "results_gpt-4o-mini_mysql.csv",
    "Qwen2.5-Coder-7B-Instruct": RESULTS_DIR / "results_qwen2.5-coder-7b_mysql.csv",
}
KEY_COLUMNS = ["dataset", "difficulty", "question", "gold_sql"]
COLORS = {"GPT-4o-mini": "#2563eb", "Qwen2.5-Coder-7B-Instruct": "#d97706"}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided 95% Wilson binomial confidence interval."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; b/c are discordant-pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    lower_tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * lower_tail)


def load_results() -> dict[str, pd.DataFrame]:
    data = {}
    required = set(KEY_COLUMNS + ["correct", "correct_lenient", "trivial_empty_match", "execution_error"])
    for model, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {model} results: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        # A trivial empty match must not contribute to the headline strict score.
        df["strict_effective"] = df["correct"].fillna(False).astype(bool) & ~df[
            "trivial_empty_match"
        ].fillna(False).astype(bool)
        df["lenient"] = df["correct_lenient"].fillna(False).astype(bool)
        df["execution_error_flag"] = df["execution_error"].notna()
        data[model] = df
    return data


def metric_rows(data: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    dimensions = [("overall", lambda df: [("All questions", df)]),
                  ("dataset", lambda df: df.groupby("dataset", sort=True)),
                  ("difficulty", lambda df: df.groupby("difficulty", sort=True))]
    for model, df in data.items():
        for dimension, groups_fn in dimensions:
            for group, subset in groups_fn(df):
                total = len(subset)
                for metric, column in [("Strict execution accuracy", "strict_effective"),
                                       ("Lenient execution accuracy", "lenient"),
                                       ("Execution error rate", "execution_error_flag")]:
                    successes = int(subset[column].sum())
                    low, high = wilson_interval(successes, total)
                    rows.append({
                        "model": model, "dimension": dimension, "group": group,
                        "metric": metric, "successes": successes, "total": total,
                        "rate": successes / total, "ci95_low": low, "ci95_high": high,
                    })
    return rows


def paired_tests(data: dict[str, pd.DataFrame]) -> list[dict]:
    gpt = data["GPT-4o-mini"].copy()
    qwen = data["Qwen2.5-Coder-7B-Instruct"].copy()

    # Public text-to-SQL corpora contain repeated identical question/SQL pairs.
    # Pair the first occurrence of each repeated key with the first occurrence
    # in the other model file, etc.  The two files are generated from the same
    # fixed sample; verify the multiplicities before making this pairing.
    gpt_counts = Counter(map(tuple, gpt[KEY_COLUMNS].itertuples(index=False, name=None)))
    qwen_counts = Counter(map(tuple, qwen[KEY_COLUMNS].itertuples(index=False, name=None)))
    if gpt_counts != qwen_counts:
        raise ValueError("GPT and Qwen files do not contain the same question multiplicities.")
    gpt["_occurrence"] = gpt.groupby(KEY_COLUMNS, dropna=False).cumcount()
    qwen["_occurrence"] = qwen.groupby(KEY_COLUMNS, dropna=False).cumcount()
    pair_keys = KEY_COLUMNS + ["_occurrence"]
    paired = gpt[pair_keys + ["strict_effective", "lenient"]].merge(
        qwen[pair_keys + ["strict_effective", "lenient"]], on=pair_keys,
        suffixes=("_gpt", "_qwen"), validate="one_to_one",
    )
    if len(paired) != len(gpt) or len(paired) != len(qwen):
        raise ValueError("GPT and Qwen files do not contain the same questions.")
    tests = []
    for column, metric in [
        ("strict_effective", "Strict execution accuracy excluding trivial empty matches"),
        ("lenient", "Lenient execution accuracy"),
    ]:
        b = int((paired[f"{column}_gpt"] & ~paired[f"{column}_qwen"]).sum())
        c = int((~paired[f"{column}_gpt"] & paired[f"{column}_qwen"]).sum())
        tests.append({
            "test": "Exact paired McNemar test (two-sided)", "metric": metric,
            "n_pairs": len(paired), "gpt_only_correct": b, "qwen_only_correct": c,
            "discordant_pairs": b + c, "p_value": exact_mcnemar_p_value(b, c),
        })
    return tests


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(path: Path, title: str, rows: list[dict], groups: list[str], metrics: list[str]) -> None:
    """Draw a grouped, labelled SVG bar chart without an external plotting dependency."""
    width, height = 1080, 620
    left, right, top, bottom = 90, 30, 88, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    model_names = list(MODEL_FILES)
    series = [(model, metric) for model in model_names for metric in metrics]
    by_key = {(r["model"], r["group"], r["metric"]): r for r in rows}
    group_width = plot_w / max(1, len(groups))
    bar_width = min(28, group_width / (len(series) + 2))
    chart = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Grouped bar chart of percentages by model and category.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="62" font-family="Arial, sans-serif" font-size="13" fill="#475569">Strict excludes trivial empty-result matches; whiskers are 95% Wilson confidence intervals.</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - (tick / 100) * plot_h
        chart.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="1"/>')
        chart.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{tick}%</text>')
    chart.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155"/>')
    chart.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#334155"/>')
    for index, group in enumerate(groups):
        centre = left + group_width * (index + 0.5)
        start_x = centre - (len(series) * bar_width) / 2
        for series_index, (model, metric) in enumerate(series):
            row = by_key.get((model, group, metric))
            if row is None:
                continue
            rate = row["rate"]
            x = start_x + series_index * bar_width
            y = top + plot_h * (1 - rate)
            h = top + plot_h - y
            opacity = "1" if metric.startswith("Strict") else "0.48"
            chart.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-3:.1f}" height="{h:.1f}" fill="{COLORS[model]}" opacity="{opacity}"/>')
            if metric.startswith("Strict"):
                low_y = top + plot_h * (1 - row["ci95_low"])
                high_y = top + plot_h * (1 - row["ci95_high"])
                mid_x = x + (bar_width - 3) / 2
                chart.append(f'<line x1="{mid_x:.1f}" y1="{low_y:.1f}" x2="{mid_x:.1f}" y2="{high_y:.1f}" stroke="#111827"/>')
                chart.append(f'<line x1="{mid_x-4:.1f}" y1="{low_y:.1f}" x2="{mid_x+4:.1f}" y2="{low_y:.1f}" stroke="#111827"/>')
                chart.append(f'<line x1="{mid_x-4:.1f}" y1="{high_y:.1f}" x2="{mid_x+4:.1f}" y2="{high_y:.1f}" stroke="#111827"/>')
            chart.append(f'<text x="{x+(bar_width-3)/2:.1f}" y="{max(top+13, y-5):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{rate:.0%}</text>')
        label = html.escape(str(group).replace("custom_", "custom "))
        chart.append(f'<text x="{centre:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{label}</text>')
    legend_x, legend_y = left, height - 43
    for i, (model, metric) in enumerate(series):
        x = legend_x + i * 235
        opacity = "1" if metric.startswith("Strict") else "0.48"
        name = "strict" if metric.startswith("Strict") else "lenient"
        chart.append(f'<rect x="{x}" y="{legend_y-11}" width="14" height="14" fill="{COLORS[model]}" opacity="{opacity}"/>')
        chart.append(f'<text x="{x+20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12">{html.escape(model)} — {name}</text>')
    chart.append('</svg>')
    path.write_text("\n".join(chart), encoding="utf-8")


def write_report(path: Path, rows: list[dict], tests: list[dict]) -> None:
    overall = [r for r in rows if r["dimension"] == "overall" and r["metric"] != "Execution error rate"]
    lines = [
        "# Statistical analysis summary",
        "",
        "## Method",
        "",
        "Accuracy intervals are 95% Wilson binomial confidence intervals. GPT and Qwen are compared using an exact two-sided paired McNemar test on the same 306 questions. A trivial empty-result match is counted as incorrect in the strict metric.",
        "",
        "## Overall results",
        "",
        "| Model | Metric | Result | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for r in overall:
        lines.append(f"| {r['model']} | {r['metric']} | {r['successes']}/{r['total']} ({r['rate']:.1%}) | {r['ci95_low']:.1%}–{r['ci95_high']:.1%} |")
    lines += ["", "## GPT vs. Qwen (paired tests)", ""]
    for test in tests:
        verdict = "statistically significant at α = 0.05" if test["p_value"] < 0.05 else "not statistically significant at α = 0.05"
        lines += [
            f"### {test['metric']}",
            "",
            f"- GPT-only correct: {test['gpt_only_correct']}",
            f"- Qwen-only correct: {test['qwen_only_correct']}",
            f"- Exact McNemar p-value: {test['p_value']:.6f}",
            f"- Conclusion: the observed difference is **{verdict}**.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    rows = metric_rows(data)
    tests = paired_tests(data)
    write_csv(OUTPUT_DIR / "metric_summary.csv", rows)
    write_csv(OUTPUT_DIR / "paired_mcnemar_test.csv", tests)
    write_report(OUTPUT_DIR / "statistical_summary.md", rows, tests)

    metrics = ["Strict execution accuracy", "Lenient execution accuracy"]
    svg_bar_chart(OUTPUT_DIR / "accuracy_overall.svg", "Overall execution accuracy", rows, ["All questions"], metrics)
    dataset_groups = sorted({r["group"] for r in rows if r["dimension"] == "dataset"})
    difficulty_order = ["easy", "medium", "hard"]
    difficulty_groups = [d for d in difficulty_order if any(r["dimension"] == "difficulty" and r["group"] == d for r in rows)]
    svg_bar_chart(OUTPUT_DIR / "accuracy_by_dataset.svg", "Execution accuracy by dataset", rows, dataset_groups, metrics)
    svg_bar_chart(OUTPUT_DIR / "accuracy_by_difficulty.svg", "Execution accuracy by difficulty", rows, difficulty_groups, metrics)
    print(f"Wrote statistical tables, summary, and charts to: {OUTPUT_DIR}")
    for test in tests:
        print(f"{test['metric']}: exact paired McNemar p={test['p_value']:.6f}")


if __name__ == "__main__":
    main()
