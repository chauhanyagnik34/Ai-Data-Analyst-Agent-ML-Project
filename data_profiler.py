"""
data_profiler.py
----------------
Module responsible for computing exploratory data analysis (EDA) and profiling summaries:
- Identifies numerical vs. categorical vs. datetime columns
- Computes missing values and duplicate rows
- Calculates descriptive statistics: mean, median, min, max, std
- Summarizes categorical columns (cardinality, most frequent values)
- Formats insights in beginner-friendly, clean formats
"""

import pandas as pd
import numpy as np


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Classifies DataFrame columns into numerical, categorical, and datetime types.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Lists of column names categorized by type.
    """
    if df is None or df.empty:
        return {"numerical": [], "categorical": [], "datetime": []}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    return {
        "numerical": num_cols,
        "categorical": cat_cols,
        "datetime": date_cols,
    }


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing (NaN/null) values count and percentage for each column.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Table with columns ['Column', 'Missing Count', 'Missing Percentage (%)'].
    """
    if df is None or df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows * 100).round(2)

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing Percentage (%)": missing_percentages.values,
    })

    # Sort so columns with highest missing values appear on top
    summary_df = summary_df.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)
    return summary_df


def get_duplicate_summary(df: pd.DataFrame) -> dict:
    """
    Calculates total duplicate rows and their percentage in the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: {'duplicate_count': int, 'duplicate_percentage': float}
    """
    if df is None or df.empty:
        return {"duplicate_count": 0, "duplicate_percentage": 0.0}

    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def get_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes key descriptive statistics for all numerical columns:
    - Count: Number of non-null records
    - Mean: Mathematical average
    - Std Dev: How spread out the values are from the average
    - Min: Smallest value
    - Median (50%): Middle value when sorted (unaffected by extreme outliers)
    - Max: Largest value

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary table of numerical statistics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        return pd.DataFrame(columns=["Column", "Count", "Mean", "Std Dev", "Min", "Median", "Max"])

    records = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        records.append({
            "Column": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 2),
            "Std Dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "Min": round(float(series.min()), 2),
            "Median": round(float(series.median()), 2),
            "Max": round(float(series.max()), 2),
        })

    return pd.DataFrame(records)


def get_categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics for categorical columns:
    - Unique Count: Number of distinct categories
    - Top Category: The most frequently occurring value (mode)
    - Frequency: How many times the top category appears

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary table of categorical statistics.
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    if len(cat_cols) == 0:
        return pd.DataFrame(columns=["Column", "Unique Count", "Top Category", "Frequency"])

    records = []
    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        unique_cnt = int(series.nunique())
        mode_val = series.mode()[0] if not series.empty else "N/A"
        freq = int((series == mode_val).sum()) if not series.empty else 0

        records.append({
            "Column": col,
            "Unique Count": unique_cnt,
            "Top Category": str(mode_val),
            "Frequency": freq,
        })

    return pd.DataFrame(records)


def generate_full_profile(df: pd.DataFrame) -> dict:
    """
    Aggregates all profiling components into a unified profile dictionary.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Complete profiling report containing DataFrames and metrics.
    """
    if df is None or df.empty:
        return {}

    col_types = get_column_types(df)
    missing_df = get_missing_summary(df)
    duplicate_info = get_duplicate_summary(df)
    num_stats = get_numerical_statistics(df)
    cat_stats = get_categorical_statistics(df)

    total_cells = df.size
    total_missing_cells = int(df.isnull().sum().sum())
    overall_missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": total_cells,
        "column_types": col_types,
        "duplicate_info": duplicate_info,
        "total_missing_cells": total_missing_cells,
        "overall_missing_percentage": overall_missing_pct,
        "missing_summary_table": missing_df,
        "numerical_stats_table": num_stats,
        "categorical_stats_table": cat_stats,
    }


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    import os

    sample_path = os.path.join("datasets", "sample_data.csv")
    df, err = load_dataset(sample_path)

    if err:
        print(f"Error loading dataset: {err}")
    else:
        print("=" * 70)
        print("DATA PROFILER REPORT")
        print("=" * 70)

        profile = generate_full_profile(df)
        print(f"Rows: {profile['total_rows']} | Columns: {profile['total_columns']}")
        print(f"Total Cells: {profile['total_cells']}")
        print(f"Duplicate Rows: {profile['duplicate_info']['duplicate_count']} ({profile['duplicate_info']['duplicate_percentage']}%)")
        print(f"Total Missing Values: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)")

        print("\n--- Column Types ---")
        print(f"Numerical: {profile['column_types']['numerical']}")
        print(f"Categorical: {profile['column_types']['categorical']}")

        print("\n--- Numerical Statistics ---")
        print(profile['numerical_stats_table'].to_string(index=False))

        print("\n--- Categorical Statistics ---")
        print(profile['categorical_stats_table'].to_string(index=False))

        print("\n--- Missing Values Summary ---")
        print(profile['missing_summary_table'].to_string(index=False))
        print("\nSUCCESS: data_profiler.py completed profiling accurately!")
