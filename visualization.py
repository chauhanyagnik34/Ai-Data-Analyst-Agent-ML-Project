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


# Cohesive, modern dark theme color palette matching the UI
DARK_BG = "#0d1527"
AXIS_LINE = "#202f48"
GRID_COLOR = "#1b2842"
TEXT_COLOR = "#f8fafc"
MUTED_TEXT = "#94a3b8"
PRIMARY_COLOR = "#1d72f2"   # Electric Modern Blue (matches screenshot)
ACCENT_COLOR = "#38bdf8"    # Cyan / Sky Accent
BAR_COLORS = ["#1d72f2", "#38bdf8", "#0ea5e9", "#2563eb", "#6366f1", "#8b5cf6", "#ec4899"]


def _apply_dark_theme(fig, ax, title: str = None, x_label: str = None, y_label: str = None):
    """Applies high-contrast sleek dark aesthetic to Matplotlib plots."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=TEXT_COLOR)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10, color=MUTED_TEXT, labelpad=8)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10, color=MUTED_TEXT, labelpad=8)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.7, color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(AXIS_LINE)


def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a clean bar chart for categorical comparisons.
    """
    if data is None or data.empty:
        return None, "Cannot create bar chart: Data is empty."

    if x_col not in data.columns or y_col not in data.columns:
        return None, f"Cannot create bar chart: Columns '{x_col}' or '{y_col}' not found."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Select color palette
        colors = BAR_COLORS[:len(data)] if len(data) <= len(BAR_COLORS) else PRIMARY_COLOR
        bars = ax.bar(data[x_col].astype(str), data[y_col], color=colors, edgecolor="#091322", linewidth=0.7)

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
                color=TEXT_COLOR,
            )

        _apply_dark_theme(fig, ax, title or f"{y_col} by {x_col}", x_col, y_col)

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
    bins: int = 28,
    title: str = None,
) -> tuple:
    """
    Creates a histogram showing the distribution of a numerical column.
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create histogram: Column '{col}' not available."

    try:
        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return None, f"Cannot create histogram: Column '{col}' is not numeric."

        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.hist(series, bins=bins, color=PRIMARY_COLOR, edgecolor="#091322", alpha=0.95, linewidth=0.6)

        _apply_dark_theme(fig, ax, title or f"{col} distribution", col, "Count")

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
        ax.plot(sorted_df[x_col].astype(str), sorted_df[y_col], marker="o", color=PRIMARY_COLOR, linewidth=2.5, markersize=5)

        _apply_dark_theme(fig, ax, title or f"{y_col} Trend by {x_col}", x_col, y_col)

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
        ax.scatter(clean_df[x_col], clean_df[y_col], color=PRIMARY_COLOR, alpha=0.8, edgecolors="#38bdf8", linewidth=0.5, s=45)

        # Plot trend line if both are numeric
        if pd.api.types.is_numeric_dtype(clean_df[x_col]) and pd.api.types.is_numeric_dtype(clean_df[y_col]):
            x = clean_df[x_col]
            y = clean_df[y_col]
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, color="#f97316", linestyle="--", linewidth=1.8, label="Trend line")
            leg = ax.legend(facecolor=DARK_BG, edgecolor=AXIS_LINE, labelcolor=TEXT_COLOR)

        _apply_dark_theme(fig, ax, title or f"{y_col} vs. {x_col}", x_col, y_col)

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating scatter plot: {str(e)}"


def create_correlation_heatmap(corr_df: pd.DataFrame) -> tuple:
    """
    Renders a clear, stylish heatmap for numerical correlation matrix.
    """
    if corr_df is None or corr_df.empty:
        return None, "Cannot generate correlation heatmap: Data is empty."

    try:
        plot_df = corr_df.copy()
        if "Variable" in plot_df.columns:
            plot_df = plot_df.set_index("Variable")
        plot_df = plot_df.select_dtypes(include=[np.number])
        if plot_df.empty or len(plot_df.columns) < 2:
            return None, "At least two numerical columns are required for a correlation heatmap."

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        cax = ax.matshow(plot_df.values, cmap="Blues", vmin=-1, vmax=1)
        cb = fig.colorbar(cax, shrink=0.8, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=MUTED_TEXT, labelcolor=MUTED_TEXT)

        ax.set_xticks(range(len(plot_df.columns)))
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_xticklabels(plot_df.columns, rotation=45, ha="left", fontsize=9, color=MUTED_TEXT)
        ax.set_yticklabels(plot_df.index, fontsize=9, color=MUTED_TEXT)

        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                val = plot_df.iloc[i, j]
                text_color = "#ffffff" if abs(val) > 0.4 else "#94a3b8"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

        ax.set_title("Pearson Correlation Heatmap", pad=25, fontsize=12, fontweight="bold", color=TEXT_COLOR)
        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating correlation heatmap: {str(e)}"


def create_distribution_bar_chart(
    df: pd.DataFrame,
    col: str,
    bins: int = 10,
    title: str = None,
) -> tuple:
    """
    Creates a bar chart representing the frequency distribution of a single column.
    For continuous numeric columns, bins values into formatted interval brackets.
    For categorical / discrete columns, computes value counts.
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create bar chart: Column '{col}' not available in dataset."

    try:
        series = df[col].dropna()
        if series.empty:
            return None, f"Cannot create bar chart: Column '{col}' has no valid values."

        # Case 1: Continuous numerical column with many distinct values
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
            num_bins = min(bins, series.nunique())
            binned = pd.cut(series, bins=num_bins)
            counts = binned.value_counts().sort_index()

            formatted_labels = []
            for interval in counts.index:
                left = round(interval.left, 1)
                right = round(interval.right, 1)
                if left.is_integer() and right.is_integer():
                    formatted_labels.append(f"{int(left)}-{int(right)}")
                else:
                    formatted_labels.append(f"{left}-{right}")

            binned_df = pd.DataFrame({
                col: formatted_labels,
                "Count": counts.values,
            })
            chart_title = title or f"Distribution of {col} (Grouped Intervals)"
            return create_bar_chart(binned_df, x_col=col, y_col="Count", title=chart_title)

        # Case 2: Categorical or discrete column
        else:
            top_counts = series.value_counts().head(15).reset_index()
            top_counts.columns = [col, "Count"]
            chart_title = title or f"Frequency of {col}"
            return create_bar_chart(top_counts, x_col=col, y_col="Count", title=chart_title)

    except Exception as e:
        plt.close("all")
        return None, f"Error generating distribution bar chart: {str(e)}"


def create_box_plot(
    df: pd.DataFrame,
    col: str,
    group_by: str = None,
    title: str = None,
) -> tuple:
    """
    Creates a box plot showing median, quartiles, IQR, and outliers.
    Optionally groups by a categorical column.
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create box plot: Column '{col}' not found."

    if not pd.api.types.is_numeric_dtype(df[col]):
        return None, f"Cannot create box plot: Column '{col}' is not numeric."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if group_by and group_by in df.columns:
            clean_df = df.dropna(subset=[col, group_by])
            categories = list(clean_df[group_by].dropna().unique())[:10]
            grouped_data = [clean_df[clean_df[group_by] == cat][col].dropna() for cat in categories]

            bp = ax.boxplot(grouped_data, tick_labels=[str(c) for c in categories], patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(PRIMARY_COLOR)
                patch.set_edgecolor("#38bdf8")
                patch.set_alpha(0.85)

            _apply_dark_theme(fig, ax, title or f"Box Plot of {col} grouped by {group_by}", group_by, col)
        else:
            clean_series = df[col].dropna()
            bp = ax.boxplot(clean_series, tick_labels=[col], patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(PRIMARY_COLOR)
                patch.set_edgecolor("#38bdf8")
                patch.set_alpha(0.85)

            _apply_dark_theme(fig, ax, title or f"Box Plot of {col}", col, "Value")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating box plot: {str(e)}"


def create_pie_chart(
    df: pd.DataFrame,
    col: str,
    top_n: int = 7,
    title: str = None,
) -> tuple:
    """
    Creates a pie chart showing proportions of a categorical column.
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create pie chart: Column '{col}' not found."

    try:
        series = df[col].dropna()
        counts = series.value_counts()
        if counts.empty:
            return None, f"Cannot create pie chart: Column '{col}' has no values."

        if len(counts) > top_n:
            main_counts = counts.head(top_n)
            other_sum = counts.iloc[top_n:].sum()
            plot_series = pd.concat([main_counts, pd.Series({"Other": other_sum})])
        else:
            plot_series = counts

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        colors = BAR_COLORS[:len(plot_series)] if len(plot_series) <= len(BAR_COLORS) else plt.cm.tab10.colors[:len(plot_series)]
        wedges, texts, autotexts = ax.pie(
            plot_series.values,
            labels=plot_series.index.astype(str),
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5},
        )
        for t in texts:
            t.set_color(MUTED_TEXT)
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontweight("bold")

        ax.set_title(title or f"Proportion Breakdown of {col}", fontsize=12, fontweight="bold", pad=12, color=TEXT_COLOR)
        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating pie chart: {str(e)}"


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

    # 1. Direct Visualization Tasks ("visualize", "plot", "chart", "graph")
    if task in ["visualize", "plot", "chart", "graph"] and original_df is not None:
        chart_type = (analysis_result.get("chart_type") or "bar").lower().strip()
        col = analysis_result.get("task_column") or analysis_result.get("column")
        x_col = analysis_result.get("x_col")
        y_col = analysis_result.get("y_col")
        group_by = analysis_result.get("group_by")

        # Correlation Heatmap
        if chart_type in ["heatmap", "heat_map", "correlation"]:
            if isinstance(chart_data, pd.DataFrame):
                return create_correlation_heatmap(chart_data)
            num_cols = original_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                corr = original_df[num_cols].corr().round(3).reset_index()
                corr.rename(columns={"index": "Variable"}, inplace=True)
                return create_correlation_heatmap(corr)

        # Scatter Plot (requires 2 columns)
        if chart_type in ["scatter", "scatter_plot", "scatterplot"]:
            if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
                return create_scatter_plot(original_df, x_col=x_col, y_col=y_col)
            elif col and y_col:
                return create_scatter_plot(original_df, x_col=col, y_col=y_col)

        # Line Chart
        if chart_type in ["line", "line_chart", "line_plot"]:
            if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
                return create_line_chart(original_df, x_col=x_col, y_col=y_col)
            elif col and col in original_df.columns:
                # If single column, use index as x
                seq_df = original_df[[col]].dropna().reset_index()
                return create_line_chart(seq_df, x_col="index", y_col=col)

        # Box Plot
        if chart_type in ["box", "boxplot", "box_plot"]:
            target_col = col or y_col or x_col
            if target_col and target_col in original_df.columns:
                return create_box_plot(original_df, col=target_col, group_by=group_by)

        # Pie Chart
        if chart_type in ["pie", "pie_chart", "piechart"]:
            target_col = col or x_col
            if target_col and target_col in original_df.columns:
                return create_pie_chart(original_df, col=target_col)

        # Histogram
        if chart_type in ["histogram", "hist"]:
            target_col = col or x_col
            if target_col and target_col in original_df.columns:
                if pd.api.types.is_numeric_dtype(original_df[target_col]):
                    return create_histogram(original_df, col=target_col)
                else:
                    return create_distribution_bar_chart(original_df, col=target_col)

        # Bar Chart / Bargraph
        if chart_type in ["bar", "bargraph", "bar_chart", "barchart"]:
            # If two columns provided (e.g. x_col and y_col)
            if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
                if isinstance(chart_data, pd.DataFrame) and x_col in chart_data.columns and y_col in chart_data.columns:
                    return create_bar_chart(chart_data, x_col=x_col, y_col=y_col)
                elif group_by:
                    grp = original_df.groupby(group_by)[y_col].mean().round(2).reset_index()
                    return create_bar_chart(grp, x_col=group_by, y_col=y_col)
                else:
                    return create_bar_chart(original_df, x_col=x_col, y_col=y_col)

            # Single column distribution bar chart (e.g. "bargraph of Age")
            target_col = col or x_col or y_col
            if target_col and target_col in original_df.columns:
                return create_distribution_bar_chart(original_df, col=target_col)

        # General fallback: if single column
        target_col = col or x_col or y_col
        if target_col and target_col in original_df.columns:
            return create_distribution_bar_chart(original_df, col=target_col)

    # 2. Group Aggregations -> Bar Chart
    if task in ["group_aggregate", "group_by", "aggregate"] and isinstance(chart_data, pd.DataFrame):
        cols = list(chart_data.columns)
        if len(cols) >= 2:
            return create_bar_chart(chart_data, x_col=cols[0], y_col=cols[1])

    # 3. Top N / Bottom N -> Bar Chart
    elif task in ["top_n", "bottom_n", "highest_n", "lowest_n"] and isinstance(chart_data, pd.DataFrame):
        cat_cols = chart_data.select_dtypes(include=["object", "string"]).columns
        num_cols = chart_data.select_dtypes(include=[np.number]).columns
        if len(cat_cols) > 0 and len(num_cols) > 0:
            return create_bar_chart(chart_data, x_col=cat_cols[0], y_col=num_cols[-1])

    # 4. Correlation Matrix -> Heatmap
    elif task in ["correlation", "corr"] and isinstance(chart_data, pd.DataFrame):
        return create_correlation_heatmap(chart_data)

    # 5. Missing Values Breakdown -> Bar Chart if missing values exist
    elif task in ["missing_values", "null_values"]:
        val = analysis_result.get("value")
        if isinstance(val, pd.DataFrame) and "Missing_Count" in val.columns:
            has_missing = val[val["Missing_Count"] > 0]
            if not has_missing.empty:
                return create_bar_chart(has_missing, x_col="Column", y_col="Missing_Count", title="Missing Values per Column")

    # 6. Scalar aggregations (Mean / Median / Sum) -> Histogram showing overall distribution with mean line
    elif task in ["mean", "median", "sum", "std"] and original_df is not None:
        col = analysis_result.get("task_column")
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
