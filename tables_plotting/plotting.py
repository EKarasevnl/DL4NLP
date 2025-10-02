from .helpers import load_all_data
#!/usr/bin/env python3
"""
plot_quant_heatmaps.py

Produces heat maps of percent-change vs Baseline for quantized MT models.
Saves PNG and SVG files to ./plots/<metric>/heat_map_<benchmark>_<quant>.png/svg

Usage:
    python plot_quant_heatmaps.py --csv example.csv

Author: (generated)
"""

from typing import List, Tuple
import os
import argparse
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---- Configuration: language order requested by user ----
LANG_ORDER = [
    "English", "Russian", "French", "German", "Dutch",
    "Polish", "Latvian", "Zulu", "Telugu", "Swahili"
]


# -------------------- Utility / Core functions --------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with original columns (expects columns like source_lang, target_lang,
        quant_level, benchmark and metric columns such as bleu, chrf, comet22, kiwi23).
    """
    df = pd.read_csv(csv_path)
    return df


def compute_percent_change_vs_baseline(
    df: pd.DataFrame,
    metric: str,
    benchmark: str,
    quant_level: str
) -> pd.DataFrame:
    """
    Compute percent change for a given metric, benchmark and quant_level relative to Baseline.

    Percent change formula used:
        percent = 100 * (quant_value - baseline_value) / baseline_value

    Missing baseline or quant entries will produce NaN.

    Args:
        df: DataFrame containing rows with at least:
            ['source_lang', 'target_lang', 'quant_level', 'benchmark', metric]
        metric: Name of metric column to compare (e.g. 'bleu', 'chrf', 'comet22', 'kiwi23').
        benchmark: Benchmark name to filter on (e.g. 'FloRes', 'WMT24++').
        quant_level: Quantization level to compute (e.g. '8-Bit', '4-Bit').

    Returns:
        pivoted DataFrame indexed by source_lang, columns target_lang, values = percent change.
        Languages not present will appear as missing (NaN).
    """
    sub = df[df['benchmark'] == benchmark]
    baseline = sub[sub['quant_level'].str.lower() == 'baseline'][['source_lang', 'target_lang', metric]]
    quant = sub[sub['quant_level'] == quant_level][['source_lang', 'target_lang', metric]]

    # rename for merge
    baseline = baseline.rename(columns={metric: 'baseline_value'})
    quant = quant.rename(columns={metric: 'quant_value'})

    merged = pd.merge(baseline, quant, on=['source_lang', 'target_lang'], how='outer')

    # Compute percent change where baseline exists (otherwise NaN)
    merged['percent_change'] = 100.0 * (merged['quant_value'] - merged['baseline_value']) / merged['baseline_value']

    # pivot to matrix
    pivot = merged.pivot(index='source_lang', columns='target_lang', values='percent_change')

    # ensure rows/cols include all languages in LANG_ORDER (so ordering later is consistent)
    return pivot


def prepare_matrix_for_plot(
    pivot: pd.DataFrame,
    languages_order: List[str]
) -> pd.DataFrame:
    """
    Reindex pivot table to ensure the requested language order on rows and columns.
    If a language is missing entirely, it will be inserted with all-NaN.

    Args:
        pivot: DataFrame from compute_percent_change_vs_baseline
        languages_order: desired order for rows and columns

    Returns:
        DataFrame reindexed to languages_order x languages_order
    """
    # reindex rows and cols
    pivot_re = pivot.reindex(index=languages_order, columns=languages_order)
    return pivot_re


def make_diverging_cmap() -> LinearSegmentedColormap:
    """
    Create a diverging colormap that maps decreases -> red, increases -> blue,
    with white near zero.

    Returns:
        Matplotlib LinearSegmentedColormap instance.
    """
    return LinearSegmentedColormap.from_list('red_white_blue', ['red', 'white', 'blue'])


def plot_heatmap_and_save(
    matrix: pd.DataFrame,
    languages_order: List[str],
    metric: str,
    benchmark: str,
    quant_level: str,
    out_png: str,
    out_svg: str,
    figsize: Tuple[int, int] = (8, 7)
) -> None:
    """
    Plot a heatmap of percent-change values and save as PNG and SVG.

    - Missing values (NaN) and diagonal (source==target) are shown as grey.
    - Negative values (decrease) appear red, positive values (increase) appear blue.
    - Color scale is centered at 0 and symmetric using the largest magnitude in the matrix.

    Args:
        matrix: DataFrame of shape (langs x langs) with percent-change values (can contain NaN).
        languages_order: same order as matrix rows/cols (used for tick labels).
        metric: metric name used for title and saving.
        benchmark: benchmark name used for title and saving.
        quant_level: quantization level string used for title and saving.
        out_png: path to save PNG.
        out_svg: path to save SVG.
        figsize: figure size in inches.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)

    arr = matrix.values.astype(float)  # may contain NaN
    # create mask for missing values
    mask_nan = np.isnan(arr)

    # create mask for diagonal (same language) -> mask as True (we'll make these grey)
    diag_mask = np.zeros(arr.shape, dtype=bool)
    for i, lang in enumerate(languages_order):
        if i < arr.shape[0] and i < arr.shape[1]:
            diag_mask[i, i] = True

    final_mask = mask_nan | diag_mask

    # compute vmin/vmax symmetrical around zero
    if np.all(np.isnan(arr)):
        max_abs = 1.0  # fallback to avoid errors
    else:
        max_abs = np.nanmax(np.abs(arr))
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1.0

    cmap = make_diverging_cmap()
    cmap.set_bad(color='lightgrey')  # color for masked (NaN + diagonal)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    # masked array so masked entries render with set_bad
    arr_ma = ma.array(arr, mask=final_mask)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr_ma, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')

    # ticks and labels
    ax.set_xticks(np.arange(len(languages_order)))
    ax.set_xticklabels(languages_order, rotation=45, ha='right')

    ax.set_yticks(np.arange(len(languages_order)))
    ax.set_yticklabels(languages_order) # reversed order
    ax.invert_yaxis()

    ax.set_xlabel("Target language", fontsize=12)
    ax.set_ylabel("Source language", fontsize=12)

    # grid lines for clarity
    ax.set_xticks(np.arange(-.5, len(languages_order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(languages_order), 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Title
    ax.set_title(f"{metric.upper()} — {quant_level} — {benchmark}")

    # colorbar (percent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percent change vs Baseline (%)', rotation=270, labelpad=15)

    # annotate cells with numeric values (optional, but helpful); show rounded values except for masked
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if final_mask[i, j]:
                continue
            val = arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    # Save both PNG and SVG
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


# -------------------- Orchestration --------------------

def make_all_heatmaps(
    csv_path: str,
    metrics: List[str],
    quant_levels: List[str],
    languages_order: List[str]
) -> None:
    """
    High-level routine to create heatmaps for all metrics, benchmarks and quant levels found in the CSV.

    For each metric:
      For each benchmark present in the CSV:
        For each quant_level in quant_levels:
          compute matrix vs Baseline and plot heatmap and save PNG + SVG into ./plots/<metric>/.

    Args:
        csv_path: path to the CSV file.
        metrics: list of metric column names to process.
        quant_levels: list of quant levels to compute (e.g. ['8-Bit','4-Bit']).
        languages_order: list of language names in the order to plot.
    """
    df = load_data(csv_path)
    benchmarks = df['benchmark'].unique().tolist()

    for metric in metrics:
        for benchmark in benchmarks:
            for q in quant_levels:
                pivot = compute_percent_change_vs_baseline(df, metric=metric, benchmark=benchmark, quant_level=q)
                matrix = prepare_matrix_for_plot(pivot, languages_order)

                # skip if matrix is entirely NaN (no baseline/quant pairs for that benchmark & quant)
                if matrix.isna().all().all():
                    # create a placeholder blank/grey plot to indicate no data (still requested to save)
                    out_png = f"./plots/heat-maps/{metric}/heat_map_{benchmark}_{q}.png"
                    out_svg = f"./plots/heat-maps/{metric}/heat_map_{benchmark}_{q}.svg"
                    # create a grey image
                    fig, ax = plt.subplots(figsize=(6,5))
                    ax.text(0.5, 0.5, "No data for this combination", ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    os.makedirs(os.path.dirname(out_png), exist_ok=True)
                    fig.savefig(out_png, dpi=300)
                    fig.savefig(out_svg)
                    plt.close(fig)
                    continue

                out_png = f"./plots/heat-maps/{metric}/heat_map_{benchmark}_{q}.png"
                out_svg = f"./plots/heat-maps/{metric}/heat_map_{benchmark}_{q}.svg"

                plot_heatmap_and_save(
                    matrix=matrix,
                    languages_order=languages_order,
                    metric=metric,
                    benchmark=benchmark,
                    quant_level=q,
                    out_png=out_png,
                    out_svg=out_svg
                )


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantization heatmaps vs Baseline.")
    parser.add_argument("--csv", default="DL4NLP/outputs/collected_results/results.csv", help="Path to CSV file with results.")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "chrf", "comet22", "kiwi23"],
                        help="Metric column names to process.")
    parser.add_argument("--quants", nargs="+", default=["8-Bit", "4-Bit"],
                        help="Quantization levels to compute (e.g. '8-Bit' '4-Bit').")
    parser.add_argument("--langs-order", nargs="+", default=LANG_ORDER,
                        help="Language order for rows/columns (space separated).")
    return parser.parse_args()


def main() -> None:
    """
    Main entrypoint for the script.
    """
    args = parse_args()
    make_all_heatmaps(csv_path=args.csv, metrics=args.metrics, quant_levels=args.quants, languages_order=args.langs_order)


if __name__ == "__main__":
    df = load_all_data()
    df.to_csv("DL4NLP/outputs/collected_results/results.csv", index=False)
    main()
