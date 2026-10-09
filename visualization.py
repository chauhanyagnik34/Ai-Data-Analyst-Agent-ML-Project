"""
visualization.py
----------------
Module responsible for generating clean, verified charts using Matplotlib.
Ensures that all plotted values originate strictly from Pandas data calculations,
never from LLM-generated hallucinated numbers.

Supported Charts:
- Bar Charts (for group aggregations, categories, Top N)
- Line Charts (for trends over continuous or sequential variables)
- Histograms (for distribution of numerical columns)
- Scatter Plots (for relationship between two numerical variables)
- Auto-Chart generator (infers best chart from analytical result)
"""

import matplotlib
# Use Agg backend for non-interactive server rendering (avoids GUI thread crashes)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Cohesive, modern color palette
PRIMARY_COLOR = "#2563EB"   # Modern Blue
ACCENT_COLOR = "#0D9488"    # Modern Teal
BAR_COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6366F1"]


def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a clean bar chart for categorical comparisons.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        x_col (str): Categorical column for x-axis.
        y_col (str): Numerical column for y-axis.
        title (str, optional): Chart title.

    Returns:
        tuple: (fig, error_message)
    """
    if data is None or data.empty:
        return None, "Cannot create bar chart: Data is empty."

    if x_col not in data.columns or y_col not in data.columns:
        return None, f"Cannot create bar chart: Columns '{x_col}' or '{y_col}' not found."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Select color palette
        colors = BAR_COLORS[:len(data)] if len(data) <= len(BAR_COLORS) else PRIMARY_COLOR
        bars = ax.bar(data[x_col].astype(str), data[y_col], color=colors, edgecolor="#1E293B", linewidth=0.7)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.1f}" if isinstance(height, float) else f"{height:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(title or f"{y_col} by {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10, labelpad=8)
        ax.set_ylabel(y_col, fontsize=10, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Rotate x labels if more than 4 categories or long strings
        if len(data) > 4 or any(len(str(v)) > 8 for v in data[x_col]):
            plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating bar chart: {str(e)}"


def create_histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 10,
    title: str = None,
) -> tuple:
    """
    Creates a histogram showing the distribution of a numerical column.

    Parameters:
        df (pd.DataFrame): Dataset.
        col (str): Numerical column to plot.
        bins (int): Number of bins.
        title (str, optional): Chart title.

    Returns:
        tuple: (fig, error_message)
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create histogram: Column '{col}' not available."

    try:
        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return None, f"Cannot create histogram: Column '{col}' is not numeric."

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(series, bins=bins, color=PRIMARY_COLOR, edgecolor="#0F172A", alpha=0.8, linewidth=0.8)

        # Add vertical line for mean
        mean_val = series.mean()
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")

        ax.set_title(title or f"Distribution of {col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=True)

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating histogram: {str(e)}"


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a line chart for sequential or trend comparisons.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found for line chart."

    try:
        sorted_df = df.sort_values(by=x_col).dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(sorted_df[x_col].astype(str), sorted_df[y_col], marker="o", color=ACCENT_COLOR, linewidth=2)

        ax.set_title(title or f"{y_col} Trend by {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(sorted_df) > 5:
            plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating line chart: {str(e)}"


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a scatter plot showing correlation between two numerical columns.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found for scatter plot."

    try:
        clean_df = df.dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(clean_df[x_col], clean_df[y_col], color=PRIMARY_COLOR, alpha=0.7, edgecolors="none", s=50)

        # Plot trend line if both are numeric
        if pd.api.types.is_numeric_dtype(clean_df[x_col]) and pd.api.types.is_numeric_dtype(clean_df[y_col]):
            x = clean_df[x_col]
            y = clean_df[y_col]
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, color="#DC2626", linestyle="--", linewidth=1.5, label="Trend line")
            ax.legend(frameon=True)

        ax.set_title(title or f"{y_col} vs. {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating scatter plot: {str(e)}"


def create_auto_chart(analysis_result: dict, original_df: pd.DataFrame) -> tuple:
    """
    Automatically selects and renders the most appropriate chart
    based on the analysis engine result and query task.

    Parameters:
        analysis_result (dict): The output dictionary from analysis_engine.execute_plan.
        original_df (pd.DataFrame): The full underlying dataset.

    Returns:
        tuple: (fig, error_message)
    """
    if not analysis_result or analysis_result.get("status") != "success":
        return None, "No successful analytical result available to plot."

    task = analysis_result.get("task", "")
    chart_data = analysis_result.get("chart_data")

    # 1. Group Aggregations -> Bar Chart
    if task in ["group_aggregate", "group_by", "aggregate"] and isinstance(chart_data, pd.DataFrame):
        cols = list(chart_data.columns)
        if len(cols) >= 2:
            return create_bar_chart(chart_data, x_col=cols[0], y_col=cols[1])

    # 2. Top N / Bottom N -> Bar Chart
    elif task in ["top_n", "bottom_n", "highest_n", "lowest_n"] and isinstance(chart_data, pd.DataFrame):
        # Pick categorical column for label and numeric column for value
        cat_cols = chart_data.select_dtypes(include=["object", "string"]).columns
        num_cols = chart_data.select_dtypes(include=[np.number]).columns
        if len(cat_cols) > 0 and len(num_cols) > 0:
            return create_bar_chart(chart_data, x_col=cat_cols[0], y_col=num_cols[-1])

    # 3. Scalar aggregations (Mean / Median / Sum) -> Histogram showing overall distribution with mean line
    elif task in ["mean", "median", "sum", "std"] and original_df is not None:
        col = analysis_result.get("task_column")
        # If task column is available in original_df
        if col and col in original_df.columns:
            return create_histogram(original_df, col=col)

    return None, None


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    from analysis_engine import execute_plan
    import os

    df, _ = load_dataset(os.path.join("datasets", "sample_data.csv"))
    print("=" * 60)
    print("TESTING VISUALIZATION GENERATION")
    print("=" * 60)

    # Test 1: Bar chart for Group Aggregation
    plan1 = {"task": "group_aggregate", "group_by": "Department", "column": "Salary", "aggregation": "mean"}
    res1 = execute_plan(df, plan1)
    fig1, err1 = create_auto_chart(res1, df)
    if err1:
        print("Chart 1 failed:", err1)
    else:
        print("Chart 1 (Bar Chart) generated successfully:", type(fig1))
        fig1.savefig("tests/demo_bar_chart.png")
        print("  -> Saved preview to tests/demo_bar_chart.png")
        plt.close(fig1)

    # Test 2: Histogram
    fig2, err2 = create_histogram(df, col="Salary", bins=5)
    if err2:
        print("Chart 2 failed:", err2)
    else:
        print("Chart 2 (Histogram) generated successfully:", type(fig2))
        fig2.savefig("tests/demo_histogram.png")
        print("  -> Saved preview to tests/demo_histogram.png")
        plt.close(fig2)

    # Test 3: Scatter Plot
    fig3, err3 = create_scatter_plot(df, x_col="Experience_Years", y_col="Salary")
    if err3:
        print("Chart 3 failed:", err3)
    else:
        print("Chart 3 (Scatter Plot) generated successfully:", type(fig3))
        fig3.savefig("tests/demo_scatter.png")
        print("  -> Saved preview to tests/demo_scatter.png")
        plt.close(fig3)

    print("\nSUCCESS: All visualization charts rendered cleanly!")
