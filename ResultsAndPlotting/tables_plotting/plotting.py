from helpers import load_all_data
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
RES_LVL_ORDER = [
    "High Resource - High Resource",
    "High Resource - Mid Resource",
    "High Resource - Low Resource",
    "Mid Resource - High Resource",
    "Mid Resource - Mid Resource",
    "Mid Resource - Low Resource",
    "Low Resource - High Resource"
    "Low Resource - Mid Resource"
    "Low Resource - Low Resource"
]

# ---- Global Color Configuration ----
# Consistent color scheme across all plots
DIRECTION_COLORS = {
    "English → Foreign": "#1f77b4",  # Blue
    "Foreign → English": "#ff7f0e",  # Orange
    "Foreign → Foreign": "#2ca02c",  # Green
    "English → English": "#d62728"   # Red (shouldn't appear in practice)
}

QUANTIZATION_COLORS = {
    "Baseline": "#3498db",           # Light Blue
    "8-Bit": "#f39c12",             # Orange
    "4-Bit": "#e74c3c"              # Red
}

PRUNING_COLORS = {
    "Baseline": "#3498db",           # Light Blue
    "10% Sparsity": "#2ecc71",      # Green
    "30% Sparsity": "#f39c12",      # Orange
    "50% Sparsity": "#e74c3c"       # Red
}

DIRECTION_MARKERS = {
    "English → Foreign": "o",
    "Foreign → English": "s", 
    "Foreign → Foreign": "^",
    "English → English": "D"
}

# ---- Helper Functions for Consistent Styling ----
def get_resource_color_mapping():
    """Get consistent color mapping for resource combinations."""
    resource_colors = {}
    cmap = plt.get_cmap('tab10')
    colors = list(cmap.colors)
    for i, res_combo in enumerate(RES_LVL_ORDER):
        resource_colors[res_combo] = colors[i % len(colors)]
    return resource_colors


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


def categorize_language_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'direction_type' column categorizing translation directions.
    
    Categories:
    - "English → Foreign": English source, non-English target
    - "Foreign → English": Non-English source, English target  
    - "Foreign → Foreign": Both source and target are non-English
    
    Args:
        df: DataFrame with 'source_lang' and 'target_lang' columns
        
    Returns:
        DataFrame with added 'direction_type' column
    """
    df = df.copy()
    
    def get_direction_type(row):
        src_is_english = row['source_lang'].lower() == 'english'
        tgt_is_english = row['target_lang'].lower() == 'english'
        
        if src_is_english and not tgt_is_english:
            return "English → Foreign"
        elif not src_is_english and tgt_is_english:
            return "Foreign → English"
        elif not src_is_english and not tgt_is_english:
            return "Foreign → Foreign"
        else:  # src_is_english and tgt_is_english (shouldn't happen in practice)
            return "English → English"
    
    df['direction_type'] = df.apply(get_direction_type, axis=1)
    return df


def clean_axis_labels(labels, chart_type="quantization"):
    """
    Clean up axis labels to be more minimal.
    
    Args:
        labels: List of labels to clean
        chart_type: "quantization" or "pruning"
        
    Returns:
        List of cleaned labels
    """
    cleaned = []
    for label in labels:
        if chart_type == "pruning":
            # Convert "30% Sparsity" to "30%"
            if "% Sparsity" in str(label):
                cleaned.append(str(label).replace("% Sparsity", "%"))
            elif "Baseline" in str(label):
                cleaned.append("Baseline")
            else:
                cleaned.append(str(label))
        else:
            # For quantization, keep as is but could be simplified if needed
            cleaned.append(str(label))
    
    return cleaned

# -------------------- Language Direction Analysis --------------------

def prepare_direction_grouped_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str
) -> pd.DataFrame:
    """
    Group data by quantization level and language direction type, compute averages.

    Args:
        df: Input DataFrame with direction_type column.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name.

    Returns:
        DataFrame with columns [quant_level, direction_type, mean_value, std_value].
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    sub = df[df["benchmark"] == benchmark].copy()
    
    grouped = (
        sub.groupby(["quant_level", "direction_type"], as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["quant_level", "direction_type", "mean_value", "std_value", "count"]

    # enforce ordering of quant levels
    quant_order = ["Baseline", "8-Bit", "4-Bit"]
    grouped["quant_level"] = pd.Categorical(grouped["quant_level"], categories=quant_order, ordered=True)
    grouped = grouped.sort_values(["direction_type", "quant_level"])
    return grouped


def prepare_direction_grouped_pruning_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str = "Pruning-WMT"
) -> pd.DataFrame:
    """
    Group pruning data by sparsity level and language direction type, compute averages.

    Args:
        df: Input DataFrame with sparsity_level column.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name (default: "Pruning-WMT").

    Returns:
        DataFrame with columns [sparsity_level, direction_type, mean_value, std_value, count].
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    sub = df[df["benchmark"] == benchmark].copy()
    
    grouped = (
        sub.groupby(["sparsity_level", "direction_type"], as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["sparsity_level", "direction_type", "mean_value", "std_value", "count"]

    # enforce ordering of sparsity levels
    sparsity_order = ["Baseline", "10% Sparsity", "30% Sparsity", "50% Sparsity"]
    grouped["sparsity_level"] = pd.Categorical(grouped["sparsity_level"], categories=sparsity_order, ordered=True)
    grouped = grouped.sort_values(["direction_type", "sparsity_level"])
    return grouped


def create_direction_comparison_overview_bar_chart(
    df: pd.DataFrame,
    metric: str,
    benchmark: str,
    out_png: str,
    chart_type: str = "quantization",  # "quantization" or "pruning"
    figsize=(10, 6)
) -> None:
    """
    Create a comprehensive bar chart showing all quantization/sparsity levels across direction types.
    
    Args:
        df: Input DataFrame.
        metric: Metric name.
        benchmark: Benchmark name.
        out_png: Path for PNG.
        chart_type: "quantization" or "pruning".
        figsize: Figure size.
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    level_col = "sparsity_level" if chart_type == "pruning" else "quant_level"
    
    # Define level order
    if chart_type == "pruning":
        level_order = ["Baseline", "10% Sparsity", "30% Sparsity", "50% Sparsity"]
    else:
        level_order = ["Baseline", "8-Bit", "4-Bit"]
    
    # Filter and group data
    sub = df[df["benchmark"] == benchmark].copy()
    
    # Group by both direction type and level
    grouped = (
        sub.groupby(["direction_type", level_col], as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["direction_type", "level", "mean_value", "std_value", "count"]
    
    # Ensure level ordering
    grouped["level"] = pd.Categorical(grouped["level"], categories=level_order, ordered=True)
    grouped = grouped.sort_values(["direction_type", "level"])
    
    # Create single plot with absolute values only
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Use global color schemes
    direction_colors = DIRECTION_COLORS
    level_colors = PRUNING_COLORS if chart_type == "pruning" else QUANTIZATION_COLORS
    
    # Get unique direction types
    direction_types = sorted(grouped["direction_type"].unique())
    n_directions = len(direction_types)
    n_levels = len(level_order)
    
    # Set up bar positions
    bar_width = 0.15
    direction_spacing = 0.8
    x_positions = np.arange(n_directions) * direction_spacing
    
    # Plot absolute values
    for i, level in enumerate(level_order):
        level_data = grouped[grouped["level"] == level]
        positions = []
        values = []
        
        for j, direction in enumerate(direction_types):
            direction_data = level_data[level_data["direction_type"] == direction]
            if not direction_data.empty:
                positions.append(x_positions[j] + (i - n_levels/2 + 0.5) * bar_width)
                values.append(direction_data["mean_value"].iloc[0])
            else:
                positions.append(x_positions[j] + (i - n_levels/2 + 0.5) * bar_width)
                values.append(0)
        
        clean_label = clean_axis_labels([level], chart_type)[0]
        ax.bar(positions, values, bar_width, label=clean_label, 
               color=level_colors[level], alpha=0.8)
    
    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    ax.set_title(f"{pretty.get(metric, metric.upper())} Scores by Direction & Level")
    ax.set_ylabel(f"{pretty.get(metric, metric.upper())} Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(direction_types, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    title_type = "Quantization" if chart_type == "quantization" else "Pruning"
    fig.suptitle(f"{benchmark} - Language Direction Overview ({title_type})", fontsize=14)
    fig.tight_layout()
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_direction_comparison_bar_chart(
    df: pd.DataFrame,
    metric: str,
    benchmark: str,
    comparison_level: str,  # e.g., "8-Bit" or "30% Sparsity"
    out_png: str,
    chart_type: str = "quantization",  # "quantization" or "pruning"
    figsize=(10, 6)
) -> None:
    """
    Create a bar chart comparing performance across direction types for a specific quantization/sparsity level.
    
    Args:
        df: Input DataFrame.
        metric: Metric name.
        benchmark: Benchmark name.
        comparison_level: Quantization level (e.g., "8-Bit") or sparsity level (e.g., "30% Sparsity").
        out_png: Path for PNG.
        chart_type: "quantization" or "pruning".
        figsize: Figure size.
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    level_col = "sparsity_level" if chart_type == "pruning" else "quant_level"
    
    # Filter data
    sub = df[
        (df["benchmark"] == benchmark) & 
        (df[level_col] == comparison_level)
    ].copy()
    
    # Group by direction type
    grouped = (
        sub.groupby("direction_type", as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["direction_type", "mean_value", "std_value", "count"]
    
    # Also get baseline for comparison
    baseline_sub = df[
        (df["benchmark"] == benchmark) & 
        (df[level_col] == "Baseline")
    ].copy()
    
    baseline_grouped = (
        baseline_sub.groupby("direction_type", as_index=False)
        .agg({metric: ["mean"]})
    )
    baseline_grouped.columns = ["direction_type", "baseline_value"]
    
    # Merge to compute percent change
    comparison_data = pd.merge(grouped, baseline_grouped, on="direction_type", how="left")
    comparison_data["percent_change"] = 100 * (
        comparison_data["mean_value"] - comparison_data["baseline_value"]
    ) / comparison_data["baseline_value"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors for direction types
    direction_colors = {
        "English → Foreign": "#1f77b4",  # blue
        "Foreign → English": "#ff7f0e",  # orange
        "Foreign → Foreign": "#2ca02c",  # green
    }

    # Plot 1: Absolute values
    colors = [direction_colors.get(dt, "#000000") for dt in comparison_data["direction_type"]]
    bars1 = ax1.bar(comparison_data["direction_type"], comparison_data["mean_value"], 
                    color=colors, alpha=0.7, width=0.6)
    
    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    ax1.set_title(f"{pretty.get(metric, metric.upper())} Scores - {comparison_level}")
    ax1.set_ylabel(f"{pretty.get(metric, metric.upper())} Score")
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Percent change vs baseline
    bars2 = ax2.bar(comparison_data["direction_type"], comparison_data["percent_change"], 
                    color=colors, alpha=0.7, width=0.6)
    
    # Add a horizontal line at 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_title(f"Change vs Baseline - {comparison_level}")
    ax2.set_ylabel("Percent Change (%)")
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value annotations on bars
    for bar, pct_change in zip(bars2, comparison_data["percent_change"]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct_change:.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9)

    fig.suptitle(f"{benchmark} - Language Direction Comparison", fontsize=13)
    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------- Line Plots --------------------
def prepare_grouped_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str
) -> pd.DataFrame:
    """
    Group data by quantization + resource level combination and compute averages.

    Args:
        df: Input DataFrame.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name.

    Returns:
        DataFrame with columns [quant_level, res_combo, mean_value].
    """
    sub = df[df["benchmark"] == benchmark].copy()
    sub["res_combo"] = (
        sub["source_lang_resource_level"] + " - " + sub["target_lang_resource_level"]
    )

    grouped = (
        sub.groupby(["quant_level", "res_combo"], as_index=False)
        .agg({metric: ["mean", "std"]})
    )
    grouped.columns = ["quant_level", "res_combo", "mean_value", "std_value"]

    # enforce ordering of quant levels
    quant_order = ["Baseline", "8-Bit", "4-Bit"]
    grouped["quant_level"] = pd.Categorical(grouped["quant_level"], categories=quant_order, ordered=True)
    grouped = grouped.sort_values(["res_combo", "quant_level"])
    return grouped


def prepare_grouped_pruning_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str = "Pruning-WMT"
) -> pd.DataFrame:
    """
    Group pruning data by sparsity level + resource level combination and compute averages.

    Args:
        df: Input DataFrame with sparsity_level column.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name (default: "Pruning-WMT").

    Returns:
        DataFrame with columns [sparsity_level, res_combo, mean_value].
    """
    sub = df[df["benchmark"] == benchmark].copy()
    sub["res_combo"] = (
        sub["source_lang_resource_level"] + " - " + sub["target_lang_resource_level"]
    )

    grouped = (
        sub.groupby(["sparsity_level", "res_combo"], as_index=False)
        .agg({metric: ["mean", "std"]})
    )
    grouped.columns = ["sparsity_level", "res_combo", "mean_value", "std_value"]

    # enforce ordering of sparsity levels
    sparsity_order = ["10% Sparsity", "30% Sparsity", "50% Sparsity"]
    grouped["sparsity_level"] = pd.Categorical(grouped["sparsity_level"], categories=sparsity_order, ordered=True)
    grouped = grouped.sort_values(["res_combo", "sparsity_level"])
    return grouped


def plot_line_chart(
    df_grouped: pd.DataFrame,
    metric: str,
    benchmark: str,
    out_png: str,
    figsize=(8, 6)
) -> None:
    """
    Plot line chart for a given benchmark × metric.

    Args:
        df_grouped: DataFrame from prepare_grouped_data().
        metric: Metric name.
        benchmark: Benchmark name.
        out_png: Path for PNG.
        : Path for SVG.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    for res_combo, group in df_grouped.groupby("res_combo"):
        color = resource_colors.get(res_combo, "#000000")  # fallback to black
        ax.plot(
            group["quant_level"],
            group["mean_value"],
            marker="o",
            label=res_combo,
            color=color
        )
        # ax.errorbar(
        #     group["quant_level"],
        #     group["mean_value"],
        #     yerr=group["std_value"],
        #     marker="o",
        #     label=res_combo,
        #     capsize=4
        # )

    ax.set_title(f"{metric.upper()} — {benchmark}")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel(f"Average {metric.upper()}")
    
    # Position legend outside the plot area to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pruning_line_chart(
    df_grouped: pd.DataFrame,
    metric: str,
    benchmark: str = "Pruning-WMT",
    out_png: str = None,
    figsize=(8, 6)
) -> None:
    """
    Plot line chart for pruning sparsity levels × metric.

    Args:
        df_grouped: DataFrame from prepare_grouped_pruning_data().
        metric: Metric name.
        benchmark: Benchmark name.
        out_png: Path for PNG.
        : Path for SVG.
    """
    if out_png is None:
        out_png = f"./plots/pruning-line-charts/{metric}/pruning_line_chart_{benchmark}.png"
        
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    for res_combo, group in df_grouped.groupby("res_combo"):
        color = resource_colors.get(res_combo, "#000000")  # fallback to black
        ax.plot(
            group["sparsity_level"],
            group["mean_value"],
            marker="o",
            label=res_combo,
            color=color
        )

    ax.set_title(f"{metric.upper()} — {benchmark} (Pruning)")
    ax.set_xlabel("Sparsity Level")
    ax.set_ylabel(f"Average {metric.upper()}")
    
    # Position legend outside the plot area to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------- Heat Maps --------------------
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
        : path to save SVG.
        figsize: figure size in inches.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

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
    plt.close(fig)


def compute_percent_change_vs_baseline_pruning(
    df: pd.DataFrame,
    metric: str,
    benchmark: str = "Pruning-WMT",
    sparsity_level: str = "10% Sparsity",
    baseline_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute percent change for pruning data relative to a baseline (typically the original unpruned model).
    Since pruning data doesn't include baseline, we need baseline data from quantization experiments.
    
    Args:
        df: DataFrame containing pruning results.
        metric: Name of metric column to compare.
        benchmark: Benchmark name for pruning (default: "Pruning-WMT").
        sparsity_level: Sparsity level to compute (e.g. '10% Sparsity').
        baseline_data: DataFrame with baseline results (from quantization data).
        
    Returns:
        pivoted DataFrame indexed by source_lang, columns target_lang, values = percent change.
    """
    pruning_sub = df[df['benchmark'] == benchmark]
    pruning_data = pruning_sub[pruning_sub['sparsity_level'] == sparsity_level][['source_lang', 'target_lang', metric]]
    
    if baseline_data is not None:
        # Use provided baseline data (should be WMT24++ baseline)
        baseline = baseline_data[
            (baseline_data['benchmark'] == 'WMT24++') & 
            (baseline_data['quant_level'] == 'Baseline')
        ][['source_lang', 'target_lang', metric]]
    else:
        # If no baseline provided, create dummy baseline (not recommended)
        print("Warning: No baseline data provided for pruning comparison")
        return pd.DataFrame()
    
    # rename for merge
    baseline = baseline.rename(columns={metric: 'baseline_value'})
    pruning_data = pruning_data.rename(columns={metric: 'pruning_value'})
    
    merged = pd.merge(baseline, pruning_data, on=['source_lang', 'target_lang'], how='outer')
    
    # Compute percent change where baseline exists
    merged['percent_change'] = 100.0 * (merged['pruning_value'] - merged['baseline_value']) / merged['baseline_value']
    
    # pivot to matrix
    pivot = merged.pivot(index='source_lang', columns='target_lang', values='percent_change')
    
    return pivot


def plot_pruning_heatmap_and_save(
    matrix: pd.DataFrame,
    languages_order: List[str],
    metric: str,
    benchmark: str = "Pruning-WMT",
    sparsity_level: str = "10% Sparsity",
    out_png: str = None,
    figsize: Tuple[int, int] = (8, 7)
) -> None:
    """
    Plot a heatmap of percent-change values for pruning and save as PNG and SVG.
    """
    if out_png is None:
        sparsity_clean = sparsity_level.replace('% Sparsity', 'pct')
        out_png = f"./plots/pruning-heat-maps/{metric}/pruning_heat_map_{benchmark}_{sparsity_clean}.png"
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

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
    ax.set_yticklabels(languages_order)
    ax.invert_yaxis()

    ax.set_xlabel("Target language", fontsize=12)
    ax.set_ylabel("Source language", fontsize=12)

    # grid lines for clarity
    ax.set_xticks(np.arange(-.5, len(languages_order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(languages_order), 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Title
    ax.set_title(f"{metric.upper()} — {sparsity_level} — {benchmark}")

    # colorbar (percent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percent change vs Baseline (%)', rotation=270, labelpad=15)

    # annotate cells with numeric values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if final_mask[i, j]:
                continue
            val = arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# -------------------- Language Direction Analysis --------------------

def prepare_direction_grouped_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str
) -> pd.DataFrame:
    """
    Group data by quantization level and language direction type, compute averages.

    Args:
        df: Input DataFrame with direction_type column.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name.

    Returns:
        DataFrame with columns [quant_level, direction_type, mean_value, std_value].
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    sub = df[df["benchmark"] == benchmark].copy()
    
    grouped = (
        sub.groupby(["quant_level", "direction_type"], as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["quant_level", "direction_type", "mean_value", "std_value", "count"]

    # enforce ordering of quant levels
    quant_order = ["Baseline", "8-Bit", "4-Bit"]
    grouped["quant_level"] = pd.Categorical(grouped["quant_level"], categories=quant_order, ordered=True)
    grouped = grouped.sort_values(["direction_type", "quant_level"])
    return grouped


def prepare_direction_grouped_pruning_data(
    df: pd.DataFrame,
    metric: str,
    benchmark: str = "Pruning-WMT"
) -> pd.DataFrame:
    """
    Group pruning data by sparsity level and language direction type, compute averages.

    Args:
        df: Input DataFrame with sparsity_level column.
        metric: Metric column to average (e.g. 'bleu').
        benchmark: Benchmark name (default: "Pruning-WMT").

    Returns:
        DataFrame with columns [sparsity_level, direction_type, mean_value, std_value, count].
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    sub = df[df["benchmark"] == benchmark].copy()
    
    grouped = (
        sub.groupby(["sparsity_level", "direction_type"], as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["sparsity_level", "direction_type", "mean_value", "std_value", "count"]

    # enforce ordering of sparsity levels
    sparsity_order = ["Baseline", "10% Sparsity", "30% Sparsity", "50% Sparsity"]
    grouped["sparsity_level"] = pd.Categorical(grouped["sparsity_level"], categories=sparsity_order, ordered=True)
    grouped = grouped.sort_values(["direction_type", "sparsity_level"])
    return grouped


def plot_direction_comparison_chart(
    df_grouped: pd.DataFrame,
    metric: str,
    benchmark: str,
    out_png: str,
    chart_type: str = "quantization",  # "quantization" or "pruning"
    figsize=(10, 6)
) -> None:
    """
    Plot line chart comparing performance across language direction types.

    Args:
        df_grouped: DataFrame from prepare_direction_grouped_data().
        metric: Metric name.
        benchmark: Benchmark name.
        out_png: Path for PNG.
        chart_type: "quantization" or "pruning" to determine x-axis.
        figsize: Figure size.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use global color schemes
    direction_colors = DIRECTION_COLORS
    direction_markers = DIRECTION_MARKERS

    x_col = "sparsity_level" if chart_type == "pruning" else "quant_level"
    x_label = "Sparsity Level" if chart_type == "pruning" else "Quantization Level"

    for direction_type, group in df_grouped.groupby("direction_type"):
        color = direction_colors.get(direction_type, "#000000")
        marker = direction_markers.get(direction_type, "o")
        
        # Simple label without count information
        label = direction_type
        
        ax.plot(
            group[x_col],
            group["mean_value"],
            marker=marker,
            color=color,
            label=label,
            linewidth=2,
            markersize=8
        )

    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    ax.set_title(f"{pretty.get(metric, metric.upper())} by Language Direction — {benchmark}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Average {pretty.get(metric, metric.upper())}")
    
    # Clean up x-axis labels to be more minimal
    if chart_type == "pruning":
        # Get current tick positions and labels
        tick_positions = ax.get_xticks()
        current_labels = [t.get_text() for t in ax.get_xticklabels()]
        cleaned_labels = clean_axis_labels(current_labels, chart_type)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(cleaned_labels)
    
    # Position legend outside the plot area to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)

    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_direction_comparison_bar_chart(
    df: pd.DataFrame,
    metric: str,
    benchmark: str,
    comparison_level: str,  # e.g., "8-Bit" or "30% Sparsity"
    out_png: str,
    chart_type: str = "quantization",  # "quantization" or "pruning"
    figsize=(12, 6)
) -> None:
    """
    Create a bar chart comparing performance across direction types for a specific quantization/sparsity level.
    
    Args:
        df: Input DataFrame.
        metric: Metric name.
        benchmark: Benchmark name.
        comparison_level: Quantization level (e.g., "8-Bit") or sparsity level (e.g., "30% Sparsity").
        out_png: Path for PNG.
        : Path for SVG.
        chart_type: "quantization" or "pruning".
        figsize: Figure size.
    """
    # Add direction categorization if not present
    if 'direction_type' not in df.columns:
        df = categorize_language_direction(df)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    level_col = "sparsity_level" if chart_type == "pruning" else "quant_level"
    
    # Filter data
    sub = df[
        (df["benchmark"] == benchmark) & 
        (df[level_col] == comparison_level)
    ].copy()
    
    # Group by direction type
    grouped = (
        sub.groupby("direction_type", as_index=False)
        .agg({metric: ["mean", "std", "count"]})
    )
    grouped.columns = ["direction_type", "mean_value", "std_value", "count"]
    
    # Also get baseline for comparison
    baseline_level = "Baseline" if chart_type == "quantization" else "Baseline"
    baseline_sub = df[
        (df["benchmark"] == benchmark) & 
        (df[level_col] == baseline_level)
    ].copy()
    
    baseline_grouped = (
        baseline_sub.groupby("direction_type", as_index=False)
        .agg({metric: ["mean"]})
    )
    baseline_grouped.columns = ["direction_type", "baseline_value"]
    
    # Merge to compute percent change
    comparison_data = pd.merge(grouped, baseline_grouped, on="direction_type", how="left")
    comparison_data["percent_change"] = 100 * (
        comparison_data["mean_value"] - comparison_data["baseline_value"]
    ) / comparison_data["baseline_value"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Use global color scheme for direction types
    direction_colors = DIRECTION_COLORS

    # Plot 1: Absolute values
    colors = [direction_colors.get(dt, "#000000") for dt in comparison_data["direction_type"]]
    bars1 = ax1.bar(comparison_data["direction_type"], comparison_data["mean_value"], 
                    color=colors, alpha=0.7, width=0.6)
    
    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    ax1.set_title(f"{pretty.get(metric, metric.upper())} Scores - {comparison_level}")
    ax1.set_ylabel(f"{pretty.get(metric, metric.upper())} Score")
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Percent change vs baseline
    bars2 = ax2.bar(comparison_data["direction_type"], comparison_data["percent_change"], 
                    color=colors, alpha=0.7, width=0.6)
    
    # Add a horizontal line at 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_title(f"Change vs Baseline - {comparison_level}")
    ax2.set_ylabel("Percent Change (%)")
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value annotations on bars
    for bar, pct_change in zip(bars2, comparison_data["percent_change"]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct_change:.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9)

    fig.suptitle(f"{benchmark} - Language Direction Comparison", fontsize=13)
    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_all_direction_comparisons(
    csv_path: str,
    metrics: List[str],
    quant_levels: List[str]
) -> None:
    """
    Generate all direction comparison charts for quantization experiments.

    Args:
        csv_path: Input CSV path.
        metrics: Metrics to process.
        quant_levels: Quantization levels to include.
    """
    df = load_data(csv_path)
    benchmarks = df["benchmark"].unique().tolist()

    for metric in metrics:
        for benchmark in benchmarks:
            # Line chart showing trends across quantization levels
            grouped = prepare_direction_grouped_data(df, metric=metric, benchmark=benchmark)
            if not grouped.empty:
                out_png = f"./plots/direction-analysis/{metric}/direction_trends_{benchmark}.png"
                plot_direction_comparison_chart(grouped, metric, benchmark, out_png, "quantization")
            
            # Overview bar chart showing all levels together
            out_png = f"./plots/direction-analysis/{metric}/direction_overview_{benchmark}.png"
            create_direction_comparison_overview_bar_chart(
                df, metric, benchmark, out_png, "quantization"
            )
                
            # Bar charts for specific quantization levels (optional - individual level details)
            for quant_level in quant_levels:
                if quant_level != "Baseline":  # Skip baseline since we compare against it
                    out_png = f"./plots/direction-analysis/{metric}/direction_bars_{benchmark}_{quant_level}.png"
                    create_direction_comparison_bar_chart(
                        df, metric, benchmark, quant_level, out_png, "quantization"
                    )


def make_all_pruning_direction_comparisons(
    pruning_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metrics: List[str],
    sparsity_levels: List[str]
) -> None:
    """
    Generate all direction comparison charts for pruning experiments.

    Args:
        pruning_df: Pruning DataFrame.
        baseline_df: Baseline DataFrame for comparison.
        metrics: Metrics to process.
        sparsity_levels: Sparsity levels to include.
    """
    # Combine pruning and baseline data for comparison
    baseline_subset = baseline_df[
        (baseline_df["benchmark"] == "WMT24++") & 
        (baseline_df["quant_level"] == "Baseline")
    ].copy()
    baseline_subset["sparsity_level"] = "Baseline"
    baseline_subset["benchmark"] = "Pruning-WMT"  # Update benchmark to match pruning data
    baseline_subset = baseline_subset.drop(columns=["quant_level"])
    
    # Ensure both dataframes have the same columns
    common_cols = list(set(pruning_df.columns) & set(baseline_subset.columns))
    pruning_subset = pruning_df[common_cols].copy()
    baseline_subset = baseline_subset[common_cols].copy()
    
    combined_df = pd.concat([baseline_subset, pruning_subset], ignore_index=True)

    # Debug: print columns to verify
    print(f"DEBUG: Combined DF columns: {list(combined_df.columns)}")
    print(f"DEBUG: Has sparsity_level: {'sparsity_level' in combined_df.columns}")
    print(f"DEBUG: Has quant_level: {'quant_level' in combined_df.columns}")

    for metric in metrics:
        # Line chart showing trends across sparsity levels
        grouped = prepare_direction_grouped_pruning_data(combined_df, metric=metric)
        if not grouped.empty:
            out_png = f"./plots/direction-analysis/{metric}/direction_trends_pruning.png"
            plot_direction_comparison_chart(grouped, metric, "Pruning-WMT", out_png, "pruning")
        
        # Overview bar chart showing all sparsity levels together
        out_png = f"./plots/direction-analysis/{metric}/direction_overview_pruning.png"
        create_direction_comparison_overview_bar_chart(
            combined_df, metric, "Pruning-WMT", out_png, "pruning"
        )
            
        # Bar charts for specific sparsity levels (optional - individual level details)
        for sparsity_level in sparsity_levels:
            out_png = f"./plots/direction-analysis/{metric}/direction_bars_pruning_{sparsity_level.replace('% Sparsity', 'pct')}.png"
            
            create_direction_comparison_bar_chart(
                combined_df, metric, "Pruning-WMT", sparsity_level, out_png, "pruning"
            )


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
                    # create a grey image
                    fig, ax = plt.subplots(figsize=(6,5))
                    ax.text(0.5, 0.5, "No data for this combination", ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    os.makedirs(os.path.dirname(out_png), exist_ok=True)
                    fig.savefig(out_png, dpi=300)
                    plt.close(fig)
                    continue

                out_png = f"./plots/heat-maps/{metric}/heat_map_{benchmark}_{q}.png"

                plot_heatmap_and_save(
                    matrix=matrix,
                    languages_order=languages_order,
                    metric=metric,
                    benchmark=benchmark,
                    quant_level=q,
                    out_png=out_png,
                )


def make_all_pruning_plots(
    pruning_csv_path: str,
    baseline_csv_path: str,
    metrics: List[str],
    sparsity_levels: List[str],
    languages_order: List[str]
) -> None:
    """
    High-level routine to create pruning heatmaps and line charts.
    
    Args:
        pruning_csv_path: path to the pruning CSV file.
        baseline_csv_path: path to the baseline CSV file (for heatmap comparison).
        metrics: list of metric column names to process.
        sparsity_levels: list of sparsity levels to compute (e.g. ['10% Sparsity']).
        languages_order: list of language names in the order to plot.
    """
    from helpers import load_pruning_data
    
    # Load pruning data
    pruning_df = load_pruning_data(pruning_csv_path.replace('/collected_results/results.csv', ''))
    
    # Load baseline data for heatmap comparison
    baseline_df = load_data(baseline_csv_path)
    
    # Generate line charts
    for metric in metrics:
        grouped = prepare_grouped_pruning_data(pruning_df, metric=metric)
        if not grouped.empty:
            plot_pruning_line_chart(grouped, metric, "Pruning-WMT")
    
    # Generate heatmaps (comparing to baseline)
    for metric in metrics:
        for sparsity in sparsity_levels:
            pivot = compute_percent_change_vs_baseline_pruning(
                pruning_df, metric=metric, sparsity_level=sparsity, baseline_data=baseline_df
            )
            if not pivot.empty:
                matrix = prepare_matrix_for_plot(pivot, languages_order)
                plot_pruning_heatmap_and_save(
                    matrix=matrix,
                    languages_order=languages_order,
                    metric=metric,
                    sparsity_level=sparsity
                )


def make_all_line_charts(
    csv_path: str,
    metrics: List[str],
    quant_levels: List[str]
) -> None:
    """
    Generate all line charts for each metric × benchmark.

    Args:
        csv_path: Input CSV path.
        metrics: Metrics to process.
        quant_levels: Quantization levels to include.
    """
    df = load_data(csv_path)
    benchmarks = df["benchmark"].unique().tolist()

    for metric in metrics:
        for benchmark in benchmarks:
            grouped = prepare_grouped_data(df, metric=metric, benchmark=benchmark)
            if grouped.empty:
                continue
            out_png = f"./plots/line-charts/{metric}/line_chart_{benchmark}.png"
            plot_line_chart(grouped, metric, benchmark, out_png, )


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantization heatmaps vs Baseline.")
    # compute a sensible default CSV path relative to the repository root (one level above this package)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(repo_root, "outputs", "collected_results", "results.csv")
    parser.add_argument("--csv", default=default_csv, help="Path to CSV file with results.")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "chrf", "comet22", "kiwi23"],
                        help="Metric column names to process.")
    parser.add_argument("--quants", nargs="+", default=["8-Bit", "4-Bit"],
                        help="Quantization levels to compute (e.g. '8-Bit' '4-Bit').")
    parser.add_argument("--langs-order", nargs="+", default=LANG_ORDER,
                        help="Language order for rows/columns (space separated).")
    return parser.parse_args()

# ---- Fix a small typo in RES_LVL_ORDER (missing commas) ----
RES_LVL_ORDER = [
    "High Resource - High Resource",
    "High Resource - Mid Resource",
    "High Resource - Low Resource",
    "Mid Resource - High Resource",
    "Mid Resource - Mid Resource",
    "Mid Resource - Low Resource",
    "Low Resource - High Resource",
    "Low Resource - Mid Resource",
    "Low Resource - Low Resource",
]

# -------------------- Main-text composite figures --------------------

def compose_bleu_heatmap_grid(out_png: str) -> None:
    """
    Create the 2x2 BLEU heatmap figure used in the main text:
    rows: FLORES-101 (top), WMT24++ (bottom)
    cols: INT8 (left), INT4 (right)
    Pulls the four existing heatmaps from ./plots/heat-maps/bleu/.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    def _show(ax, path, title):
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)

    # Top row: FLORES-101
    _show(axes[0, 0], "./plots/heat-maps/bleu/heat_map_FloRes_8-Bit.png",
          "FLORES-101, 8-bit")
    _show(axes[0, 1], "./plots/heat-maps/bleu/heat_map_FloRes_4-Bit.png",
          "FLORES-101, 4-bit")

    # Bottom row: WMT24++
    _show(axes[1, 0], "./plots/heat-maps/bleu/heat_map_WMT24++_8-Bit.png",
          "WMT24++, 8-bit")
    _show(axes[1, 1], "./plots/heat-maps/bleu/heat_map_WMT24++_4-Bit.png",
          "WMT24++, 4-bit")

    fig.suptitle("BLEU: Percent change vs. FP16 across benchmarks and precisions", fontsize=13)
    fig.tight_layout(rect=[0, 0.00, 1, 0.97])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def compose_summary_line_figure(metric: str, out_png: str,
                               heatmap1_metric: str = None, heatmap1_quant: str = "8-Bit",
                               heatmap2_metric: str = None, heatmap2_quant: str = "8-Bit") -> None:
    """
    Create the 2x2 summary figure for the main text:
    top row: line charts (FLORES-101, WMT24++)
    bottom row: heatmaps (FLORES-101, WMT24++)
    
    Args:
        metric: Metric for line charts ('bleu', 'comet22', etc.)
        out_png: Output PNG path
        : Output SVG path
        heatmap1_metric: Metric for first heatmap (FLORES-101, defaults to line chart metric)
        heatmap1_quant: Quantization level for first heatmap (default: '8-Bit')
        heatmap2_metric: Metric for second heatmap (WMT24++, defaults to line chart metric)
        heatmap2_quant: Quantization level for second heatmap (default: '8-Bit')
    """
    # Plot both line charts and heatmaps
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # Use same metric for heatmaps if not specified
    if heatmap1_metric is None:
        heatmap1_metric = metric
    if heatmap2_metric is None:
        heatmap2_metric = metric
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Load collected results (repo-root relative)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(repo_root, "outputs", "collected_results", "results.csv")
    df = load_data(csv_path)

    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    quant_order = ["Baseline", "8-Bit", "4-Bit"]

    benchmarks = [("FloRes", "FLORES-101"), ("WMT24++", "WMT24++")]

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    # Keep track of legend handles/labels in the requested RES_LVL_ORDER
    legend_handles = []
    legend_labels = []

    # Top row: Line charts
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[0, i]
        grouped = prepare_grouped_data(df, metric=metric, benchmark=bench_key)

        # Plot each resource-combo in the fixed RES_LVL_ORDER to keep consistent ordering/colors
        for j, res_combo in enumerate(RES_LVL_ORDER):
            grp = grouped[grouped['res_combo'] == res_combo]
            if grp.empty:
                continue
            # reindex to quant_order to ensure missing quant levels appear as NaN
            grp = grp.set_index('quant_level').reindex(quant_order).reset_index()
            color = resource_colors.get(res_combo, "#000000")
            line, = ax.plot(grp['quant_level'], grp['mean_value'], marker='o', label=res_combo, color=color)
            # Collect handles/labels for the legend (only once)
            if res_combo not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(res_combo)

        ax.set_title(f"{bench_title}")
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel(f"Average {pretty.get(metric, metric.upper())}")
        ax.set_ylim(bottom=None)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)

    # Bottom row: Heatmaps
    heatmap_configs = [(heatmap1_metric, heatmap1_quant), (heatmap2_metric, heatmap2_quant)]
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[1, i]
        hm_metric, hm_quant = heatmap_configs[i]
        
        # Load and display heatmap
        heatmap_path = f"./plots/heat-maps/{hm_metric}/heat_map_{bench_key}_{hm_quant}.png"
        if os.path.exists(heatmap_path):
            img = plt.imread(heatmap_path)
            ax.imshow(img)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(heatmap_path)}",
                   ha="center", va="center", fontsize=11)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        ax.set_axis_off()

    # Adjust subplot parameters for 2x2 layout
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, wspace=0.1, hspace=0.25)

    # Create compact legend at the bottom
    ncol = 3
    legend = fig.legend(legend_handles, legend_labels, 
                       title='Resource Combination', 
                       loc='lower center',
                       ncol=ncol,
                       fontsize=9, 
                       title_fontsize=10,
                       frameon=True,
                       bbox_to_anchor=(0.5, 0.02),
                       columnspacing=1.0,
                       handlelength=1.8,
                       handletextpad=0.5)

    # No main title - subplot titles are sufficient
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compose_line_charts_only(metric: str, out_png: str) -> None:
    """
    Create side-by-side line charts (FLORES-101, WMT24++) without heatmaps.
    
    Args:
        metric: Metric for line charts ('bleu', 'comet22', etc.)
        out_png: Output PNG path
        : Output SVG path
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # Create figure with taller, less wide dimensions for ACL paper format
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))

    # Load collected results (repo-root relative)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(repo_root, "outputs", "collected_results", "results.csv")
    df = load_data(csv_path)

    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    quant_order = ["Baseline", "8-Bit", "4-Bit"]

    benchmarks = [("FloRes", "FLORES-101"), ("WMT24++", "WMT24++")]

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    # Keep track of legend handles/labels in the requested RES_LVL_ORDER
    legend_handles = []
    legend_labels = []

    # Line charts
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[i]
        grouped = prepare_grouped_data(df, metric=metric, benchmark=bench_key)

        # Plot each resource-combo in the fixed RES_LVL_ORDER to keep consistent ordering/colors
        for j, res_combo in enumerate(RES_LVL_ORDER):
            grp = grouped[grouped['res_combo'] == res_combo]
            if grp.empty:
                continue
            # reindex to quant_order to ensure missing quant levels appear as NaN
            grp = grp.set_index('quant_level').reindex(quant_order).reset_index()
            color = resource_colors.get(res_combo, "#000000")
            line, = ax.plot(grp['quant_level'], grp['mean_value'], marker='o', label=res_combo, color=color)
            # Collect handles/labels for the legend (only once)
            if res_combo not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(res_combo)

        # Compact titles and labels for paper format
        ax.set_title(f"{bench_title}", fontsize=11, pad=8)
        ax.set_xlabel("Quantization Level", fontsize=10)
        ax.set_ylabel(f"Average {pretty.get(metric, metric.upper())}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.set_ylim(bottom=None)
        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.3)

    # Tight layout with minimal margins for ACL paper format
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.35, wspace=0.15)

    # Organize legend items in a logical order
    desired_order = [
        "High Resource - High Resource", "High Resource - Mid Resource", "High Resource - Low Resource",
        "Mid Resource - High Resource", "Mid Resource - Mid Resource", "Mid Resource - Low Resource", 
        "Low Resource - High Resource", "Low Resource - Mid Resource", "Low Resource - Low Resource"
    ]
    
    # Reorder legend handles and labels to follow logical grouping
    ordered_handles = []
    ordered_labels = []
    for combo in desired_order:
        if combo in legend_labels:
            idx = legend_labels.index(combo)
            ordered_handles.append(legend_handles[idx])
            ordered_labels.append(legend_labels[idx])

    # Compact legend at the bottom optimized for paper with better organization
    ncol = 3  # Change back to 3 columns for better organization
    legend = fig.legend(ordered_handles, ordered_labels, 
                       title='Resource Combination', 
                       loc='lower center',
                       ncol=ncol,
                       fontsize=8, 
                       title_fontsize=9,
                       frameon=False,  # Remove frame for cleaner look
                       bbox_to_anchor=(0.5, 0.02),
                       columnspacing=1.0,
                       handlelength=1.5,
                       handletextpad=0.4)
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def compose_heatmaps_only(heatmap1_metric: str, heatmap1_quant: str, 
                         heatmap2_metric: str, heatmap2_quant: str,
                         out_png: str) -> None:
    """
    Create side-by-side heatmaps (FLORES-101, WMT24++) without line charts.
    
    Args:
        heatmap1_metric: Metric for first heatmap (FLORES-101)
        heatmap1_quant: Quantization level for first heatmap
        heatmap2_metric: Metric for second heatmap (WMT24++)
        heatmap2_quant: Quantization level for second heatmap
        out_png: Output PNG path
        : Output SVG path
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    benchmarks = [("FloRes", "FLORES-101"), ("WMT24++", "WMT24++")]
    heatmap_configs = [(heatmap1_metric, heatmap1_quant), (heatmap2_metric, heatmap2_quant)]
    
    # Heatmaps
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[i]
        hm_metric, hm_quant = heatmap_configs[i]
        
        # Load and display heatmap
        heatmap_path = f"./plots/heat-maps/{hm_metric}/heat_map_{bench_key}_{hm_quant}.png"
        if os.path.exists(heatmap_path):
            img = plt.imread(heatmap_path)
            ax.imshow(img)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(heatmap_path)}",
                   ha="center", va="center", fontsize=11)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        ax.set_axis_off()

    # Adjust subplot parameters
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.1)
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compose_pruning_summary_figure(metric: str, out_png: str) -> None:
    """
    Create a composite figure showing pruning results:
    Top: line chart showing performance across sparsity levels
    Bottom: heatmaps for different sparsity levels side by side
    
    Args:
        metric: Metric for plots ('bleu', 'comet22', etc.)
        out_png: Output PNG path
        : Output SVG path
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    
    # Create figure with 2x2 layout (top row: line chart, bottom row: heatmaps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Load pruning data
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(repo_root, "outputs")
    
    from helpers import load_pruning_data
    pruning_df = load_pruning_data(data_path=data_path)
    
    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    sparsity_order = ["10% Sparsity", "30% Sparsity", "50% Sparsity"]

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    # Top row: Line chart (span both columns)
    axes[0, 1].remove()  # Remove the second subplot in the top row
    ax_line = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    
    grouped = prepare_grouped_pruning_data(pruning_df, metric=metric)
    
    # Keep track of legend handles/labels
    legend_handles = []
    legend_labels = []
    
    # Plot each resource-combo
    for j, res_combo in enumerate(RES_LVL_ORDER):
        grp = grouped[grouped['res_combo'] == res_combo]
        if grp.empty:
            continue
        # reindex to sparsity_order to ensure missing sparsity levels appear as NaN
        grp = grp.set_index('sparsity_level').reindex(sparsity_order).reset_index()
        color = resource_colors.get(res_combo, "#000000")
        line, = ax_line.plot(grp['sparsity_level'], grp['mean_value'], marker='o', label=res_combo, color=color)
        # Collect handles/labels for the legend
        if res_combo not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(res_combo)

    ax_line.set_title(f"Pruning Performance - {pretty.get(metric, metric.upper())}")
    ax_line.set_xlabel("Sparsity Level")
    ax_line.set_ylabel(f"Average {pretty.get(metric, metric.upper())}")
    ax_line.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)

    # Bottom row: Heatmaps (10% and 50% sparsity)
    sparsity_configs = ["30% Sparsity", "50% Sparsity"]
    sparsity_titles = ["30% Sparsity", "50% Sparsity"]
    
    for i, (sparsity_level, title) in enumerate(zip(sparsity_configs, sparsity_titles)):
        ax = axes[1, i] 
        # Load and display heatmap
        sparsity_clean = sparsity_level.replace('% Sparsity', 'pct')
        heatmap_path = f"./plots/pruning-heat-maps/{metric}/pruning_heat_map_Pruning-WMT_{sparsity_clean}.png"
        if os.path.exists(heatmap_path):
            img = plt.imread(heatmap_path)
            ax.imshow(img)
            ax.set_title(f"Pruning {title}")
        else:
            ax.text(0.5, 0.5, f"Missing:\\n{os.path.basename(heatmap_path)}",
                   ha="center", va="center", fontsize=11)
            ax.set_title(f"Pruning {title}")
        ax.set_axis_off()

    # Adjust subplot parameters
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, wspace=0.1, hspace=0.25)

    # Create compact legend at the bottom
    ncol = 3
    legend = fig.legend(legend_handles, legend_labels, 
                       title='Resource Combination', 
                       loc='lower center',
                       ncol=ncol,
                       fontsize=9, 
                       title_fontsize=10,
                       frameon=True,
                       bbox_to_anchor=(0.5, 0.02),
                       columnspacing=1.0,
                       handlelength=1.8,
                       handletextpad=0.5)
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compose_pruning_heatmaps_only(sparsity1: str, sparsity2: str, 
                                 metric1: str, metric2: str,
                                 out_png: str) -> None:
    """
    Create side-by-side heatmaps for pruning results.
    
    Args:
        sparsity1: Sparsity level for first heatmap (e.g. "10% Sparsity")
        sparsity2: Sparsity level for second heatmap (e.g. "50% Sparsity")
        metric1: Metric for first heatmap
        metric2: Metric for second heatmap
        out_png: Output PNG path
        : Output SVG path
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    heatmap_configs = [(sparsity1, metric1), (sparsity2, metric2)]
    
    # Heatmaps
    for i, (sparsity_level, metric) in enumerate(heatmap_configs):
        ax = axes[i]
        
        # Load and display heatmap
        sparsity_clean = sparsity_level.replace('% Sparsity', 'pct')
        heatmap_path = f"./plots/pruning-heat-maps/{metric}/pruning_heat_map_Pruning-WMT_{sparsity_clean}.png"
        if os.path.exists(heatmap_path):
            img = plt.imread(heatmap_path)
            ax.imshow(img)
            ax.set_title(f"Pruning {sparsity_level} - {metric.upper()}")
        else:
            ax.text(0.5, 0.5, f"Missing:\\n{os.path.basename(heatmap_path)}",
                   ha="center", va="center", fontsize=11)
            ax.set_title(f"Pruning {sparsity_level} - {metric.upper()}")
        ax.set_axis_off()

    # Adjust subplot parameters
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.1)
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compose_summary_line_figure_right_legend(metric: str, out_png: str, 
                                           heatmap1_metric: str = 'kiwi23', heatmap1_quant: str = "4-Bit",
                                           heatmap2_metric: str = 'kiwi23', heatmap2_quant: str = "4-Bit") -> None:
    """
    Alternative version with legend to the right of the plots.
    Creates 2x2 layout with line charts on top, heatmaps on bottom.
    
    Args:
        metric: Metric for line charts ('bleu', 'comet22', etc.)
        out_png: Output PNG path
        : Output SVG path
        heatmap1_metric: Metric for first heatmap (FLORES-101, defaults to line chart metric)
        heatmap1_quant: Quantization level for first heatmap (default: '8-Bit')
        heatmap2_metric: Metric for second heatmap (WMT24++, defaults to line chart metric)
        heatmap2_quant: Quantization level for second heatmap (default: '8-Bit')
    """
    # Plot both line charts and heatmaps
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # Use same metric for heatmaps if not specified
    if heatmap1_metric is None:
        heatmap1_metric = metric
    if heatmap2_metric is None:
        heatmap2_metric = metric
    
    # Create figure with 2x2 layout but extra width for right legend
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Load collected results (repo-root relative)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(repo_root, "outputs", "collected_results", "results.csv")
    df = load_data(csv_path)

    pretty = {"bleu": "BLEU", "chrf": "ChrF", "comet22": "COMET-22", "kiwi23": "COMET-Kiwi-23-XL"}
    quant_order = ["Baseline", "8-Bit", "4-Bit"]

    benchmarks = [("FloRes", "FLORES-101"), ("WMT24++", "WMT24++")]

    # Use consistent color mapping for resource combinations
    resource_colors = get_resource_color_mapping()

    # Keep track of legend handles/labels in the requested RES_LVL_ORDER
    legend_handles = []
    legend_labels = []

    # Top row: Line charts
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[0, i]
        grouped = prepare_grouped_data(df, metric=metric, benchmark=bench_key)

        # Plot each resource-combo in the fixed RES_LVL_ORDER to keep consistent ordering/colors
        for j, res_combo in enumerate(RES_LVL_ORDER):
            grp = grouped[grouped['res_combo'] == res_combo]
            if grp.empty:
                continue
            # reindex to quant_order to ensure missing quant levels appear as NaN
            grp = grp.set_index('quant_level').reindex(quant_order).reset_index()
            color = resource_colors.get(res_combo, "#000000")
            line, = ax.plot(grp['quant_level'], grp['mean_value'], marker='o', label=res_combo, color=color)
            # Collect handles/labels for the legend (only once)
            if res_combo not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(res_combo)

        ax.set_title(f"{bench_title}")
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel(f"Average {pretty.get(metric, metric.upper())}")
        ax.set_ylim(bottom=None)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)

    # Bottom row: Heatmaps
    heatmap_configs = [(heatmap1_metric, heatmap1_quant), (heatmap2_metric, heatmap2_quant)]
    for i, (bench_key, bench_title) in enumerate(benchmarks):
        ax = axes[1, i]
        hm_metric, hm_quant = heatmap_configs[i]
        
        # Load and display heatmap
        heatmap_path = f"./plots/heat-maps/{hm_metric}/heat_map_{bench_key}_{hm_quant}.png"
        if os.path.exists(heatmap_path):
            img = plt.imread(heatmap_path)
            ax.imshow(img)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(heatmap_path)}",
                   ha="center", va="center", fontsize=11)
            ax.set_title(f"{bench_title} - {hm_metric.upper()} {hm_quant}")
        ax.set_axis_off()

    # Adjust subplot parameters to make room for the legend on the right
    fig.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08, wspace=0.1, hspace=0.25)

    # Create vertical legend to the right of the plots
    legend = fig.legend(legend_handles, legend_labels, 
                       title='Resource Combination', 
                       loc='center right',
                       ncol=1,
                       fontsize=9, 
                       title_fontsize=10,
                       frameon=True,
                       bbox_to_anchor=(0.90, 0.5),
                       handlelength=1.8,
                       handletextpad=0.5)

    # No main title - subplot titles are sufficient
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """
    Main entrypoint for the script.
    """
    args = parse_args()

    # If the CSV doesn't exist, attempt to (re)generate it from the repo outputs/ data
    if not os.path.exists(args.csv):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        data_path = os.path.join(repo_root, "outputs")
        print(f"CSV not found at {args.csv}. Attempting to generate from data at {data_path}...")
        try:
            df = load_all_data(data_path=data_path)
            os.makedirs(os.path.dirname(args.csv), exist_ok=True)
            df.to_csv(args.csv, index=False)
            print(f"Wrote collected results to {args.csv}")
        except Exception as e:
            print(f"Failed to generate collected CSV: {e}")
            raise

    """

    make_all_heatmaps(csv_path=args.csv, metrics=args.metrics, quant_levels=args.quants, languages_order=args.langs_order)
    make_all_line_charts(
        csv_path=args.csv,
        metrics=args.metrics,
        quant_levels=["Baseline", "8-Bit", "4-Bit"]
    )

    # ---- Compose the two main-text figures ----
    os.makedirs("./plots/main", exist_ok=True)

    # Main Figure 1: BLEU heatmap grid (2x2)
    compose_bleu_heatmap_grid(
        out_png="./plots/main/main_bleu_heatmaps.png",
    )

    # Main Figure 2: summary line chart with heatmaps (choose 'bleu' or 'comet22')
    # Generate both versions - legend below and legend to the right
    compose_summary_line_figure(
        metric="comet22",
        out_png="./plots/main/main_bleu_summary.png",
    )
    
    # Alternative version with legend to the right
    compose_summary_line_figure_right_legend(
        metric="comet22",
        out_png="./plots/main/main_bleu_summary_right_legend.png",
    )
    
    # NEW: Separate line charts and heatmaps
    # Line charts only (BLEU metric)
    compose_line_charts_only(
        metric="bleu",
        out_png="./plots/main/line_charts_bleu.png",
    )
    
    # Heatmaps only (any two combinations you want)
    compose_heatmaps_only(
        heatmap1_metric="bleu", heatmap1_quant="8-Bit",
        heatmap2_metric="comet22", heatmap2_quant="4-Bit",
        out_png="./plots/main/heatmaps_custom.png",
    )
    """


def main():
    """Main function to generate all plots. Comment out sections as needed."""
    # Compute repository root (one level up from this package) and use its outputs/ directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(repo_root, "outputs")
    df = load_all_data(data_path=data_path)
    out_csv = os.path.join(data_path, "collected_results", "results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Define common metrics used across multiple plot types
    metrics = ["bleu", "comet22", "chrf", "kiwi23"]
    
    # ---- SECTION 1: COMET-22 Line Charts (ACL Paper Format) ----
    print("Generating COMET-22 line charts...")
    os.makedirs("./plots/main", exist_ok=True)
    
    compose_line_charts_only(
        metric="comet22",
        out_png="./plots/main/comet22_line_charts.png"
    )
    print("Generated COMET-22 line charts: ./plots/main/comet22_line_charts.png")

    # ---- SECTION 2: Pruning Analysis Plots ----
    print("\nGenerating pruning plots...")
    os.makedirs("./plots/pruning-line-charts", exist_ok=True)
    os.makedirs("./plots/pruning-heat-maps", exist_ok=True)
    
    # Load pruning data
    # Using direct implementation to ensure English data is included
    from pathlib import Path
    from result_processing import pruning_res_processing, pruning_flores_res_processing
    from helpers import RESOURCE_LVL_MAP
    
    extractions = []
    pruning_dir = "pruning_WMT"
    full_path = Path(os.path.join(data_path, pruning_dir))
    
    # Process multilang files (X↔Y)
    multilang_pattern = 'multilang_scores_*_pruned_s*.csv'
    multilang_files = list(full_path.glob(multilang_pattern))
    
    for file_path in multilang_files:
        raw_df = pd.read_csv(file_path, index_col=None)
        raw_df["benchmark"] = "Pruning-WMT"
        filename = str(file_path.name)
        sparsity_match = filename.split('_s')[-1].replace('.csv', '')
        sparsity_level = f"{sparsity_match}% Sparsity"
        raw_df["mode"] = sparsity_level
        processed_df = pruning_res_processing(raw_df)
        extractions.append(processed_df)
    
    # Process FLORES files (English↔X)
    flores_pattern = 'scores_pruned_s*.csv'
    flores_files = list(full_path.glob(flores_pattern))
    
    for file_path in flores_files:
        raw_df = pd.read_csv(file_path, index_col=None)
        raw_df["benchmark"] = "Pruning-WMT"
        filename = str(file_path.name)
        if 's0p' in filename:
            sparsity_match = filename.split('s0p')[-1].replace('.csv', '')
            sparsity_level = f"{sparsity_match}% Sparsity"
        raw_df["mode"] = sparsity_level
        processed_df = pruning_flores_res_processing(raw_df)
        extractions.append(processed_df)
    
    pruning_df = pd.concat(extractions, ignore_index=True)
    pruning_df["source_lang_resource_level"] = pruning_df["source_lang"].map(RESOURCE_LVL_MAP)
    pruning_df["target_lang_resource_level"] = pruning_df["target_lang"].map(RESOURCE_LVL_MAP)
    
    print(f"Loaded {len(pruning_df)} pruning data rows")
    print("Sparsity levels found:", pruning_df['sparsity_level'].unique())
    
    # Generate pruning line charts
    for metric in metrics:
        print(f"Generating pruning line chart for {metric}...")
        grouped = prepare_grouped_pruning_data(pruning_df, metric=metric)
        if not grouped.empty:
            plot_pruning_line_chart(grouped, metric, "Pruning-WMT")
            print(f"  -> Saved pruning line chart for {metric}")
    
    # Generate pruning heatmaps (comparing to baseline)
    baseline_df = load_data(out_csv)
    sparsity_levels = ["10% Sparsity", "30% Sparsity", "50% Sparsity"]
    
    for metric in metrics:
        for sparsity in sparsity_levels:
            print(f"Generating pruning heatmap for {metric} at {sparsity}...")
            pivot = compute_percent_change_vs_baseline_pruning(
                pruning_df, metric=metric, sparsity_level=sparsity, baseline_data=baseline_df
            )
            if not pivot.empty:
                matrix = prepare_matrix_for_plot(pivot, LANG_ORDER)
                plot_pruning_heatmap_and_save(
                    matrix=matrix,
                    languages_order=LANG_ORDER,
                    metric=metric,
                    sparsity_level=sparsity
                )
                print(f"  -> Saved pruning heatmap for {metric} at {sparsity}")
            else:
                print(f"  -> No data for {metric} at {sparsity}")
    
    print("\nPruning line charts and heatmaps generated successfully!")
    print("- Line charts: ./plots/pruning-line-charts/")
    print("- Heatmaps: ./plots/pruning-heat-maps/")

    # ---- SECTION 3: Direction Analysis Plots ----
    print("\nGenerating language direction analysis plots...")
    
    # Create directory for direction analysis
    os.makedirs("./plots/direction-analysis", exist_ok=True)
    
    # Generate direction comparison plots for quantization data
    quant_levels = ["8-Bit", "4-Bit"]
    
    make_all_direction_comparisons(
        csv_path=out_csv,
        metrics=metrics,
        quant_levels=quant_levels
    )
    
    # Generate direction comparison plots for pruning data
    make_all_pruning_direction_comparisons(
        pruning_df=pruning_df,
        baseline_df=baseline_df,
        metrics=metrics,
        sparsity_levels=sparsity_levels
    )
    
    print("Generated direction analysis plots:")
    print("- Quantization direction trends: ./plots/direction-analysis/{metric}/direction_trends_{benchmark}.png")
    print("- Quantization direction bars: ./plots/direction-analysis/{metric}/direction_bars_{benchmark}_{level}.png")
    print("- Pruning direction trends: ./plots/direction-analysis/{metric}/direction_trends_pruning.png")
    print("- Pruning direction bars: ./plots/direction-analysis/{metric}/direction_bars_pruning_{level}.png")

    # ---- SECTION 4: Composite Pruning Figures ----
    print("\nGenerating composite pruning figures...")
    
    # Create pruning summary figures for different metrics
    compose_pruning_summary_figure(
        metric="bleu",
        out_png="./plots/main/pruning_summary_bleu.png",
    )
    
    compose_pruning_summary_figure(
        metric="comet22",
        out_png="./plots/main/pruning_summary_comet22.png",
    )
    
    # Create side-by-side pruning heatmaps
    compose_pruning_heatmaps_only(
        sparsity1="10% Sparsity", metric1="bleu",
        sparsity2="50% Sparsity", metric2="bleu",
        out_png="./plots/main/pruning_heatmaps_bleu_10vs50.png",
    )
    
    compose_pruning_heatmaps_only(
        sparsity1="30% Sparsity", metric1="comet22",
        sparsity2="50% Sparsity", metric2="kiwi23",
        out_png="./plots/main/pruning_heatmaps_mixed.png",
    )
    
    print("Generated composite pruning figures:")
    print("- Summary figures: pruning_summary_bleu.png, pruning_summary_comet22.png")
    print("- Heatmap comparisons: pruning_heatmaps_bleu_10vs50.png, pruning_heatmaps_mixed.png")

    # ---- SECTION 5: Quantization Line Charts ----
    print("\nGenerating quantization line charts...")
    make_all_line_charts(
        csv_path=out_csv,
        metrics=metrics,
        quant_levels=["Baseline", "8-Bit", "4-Bit"]
    )
    print("Generated quantization line charts: ./plots/line-charts/")

    # ---- SECTION 6: Quantization Heatmaps ----
    print("\nGenerating quantization heatmaps...")
    make_all_heatmaps(
        csv_path=out_csv,
        metrics=metrics,
        quant_levels=["8-Bit", "4-Bit"],
        languages_order=LANG_ORDER
    )
    print("Generated quantization heatmaps: ./plots/heat-maps/")


if __name__ == "__main__":
    main()
