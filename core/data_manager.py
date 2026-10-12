"""
data_loader.py
--------------
Module responsible for safely reading and validating tabular datasets (CSV and Excel).
Follows best practices:
- Validates file existence and supported extensions (.csv, .xlsx, .xls)
- Gracefully handles empty, missing, or malformed files without crashing
- Standardizes column names (strips accidental spaces)
- Computes basic dataset metadata for display in the UI
"""

import os
import pandas as pd


# Allowed file extensions for tabular uploads
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# Maximum allowed file size in bytes (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


def load_dataset(file_path: str, sheet_name=None):
    """
    Safely loads a CSV or Excel file into a Pandas DataFrame.

    Parameters:
        file_path (str): Full path to the uploaded file.
        sheet_name (str or int, optional): Excel sheet name or index to load.

    Returns:
        tuple: (df, error_message)
               - df: pd.DataFrame if successful, else None
               - error_message: str describing failure, or None if successful
    """
    # 1. Check if path is provided
    if not file_path:
        return None, "Error: No file path provided."

    # 2. Check file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            None,
            f"Error: Unsupported format '{extension}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.",
        )

    # 3. Check if file physically exists on disk
    if not os.path.exists(file_path):
        return None, f"Error: File not found at '{file_path}'."

    # 4. Enforce file size limit to prevent Denial-of-Service / Memory exhaustion
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return None, f"Error: File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 25 MB."

    # 5. Attempt to load the data using Pandas with encoding fallback
    try:
        if extension == ".csv":
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None
            last_err = None
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError as ue:
                    last_err = ue
                    continue
            if df is None:
                return None, f"Error decoding CSV file across supported encodings: {last_err}"
        else:
            # Excel file loading
            with pd.ExcelFile(file_path) as excel_file:
                target_sheet = sheet_name if sheet_name is not None else excel_file.sheet_names[0]
                df = pd.read_excel(excel_file, sheet_name=target_sheet)

        # 6. Check if dataset is empty
        if df is None or df.empty or len(df.columns) == 0:
            return None, "Error: The uploaded file is empty (contains 0 rows or 0 columns)."

        # 7. Clean column names (strip leading/trailing whitespace) and deduplicate
        cleaned_cols = []
        counts = {}
        for col in df.columns:
            name = str(col).strip() if str(col).strip() else "Unnamed"
            if name in counts:
                counts[name] += 1
                cleaned_cols.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                cleaned_cols.append(name)
        df.columns = cleaned_cols

        return df, None

    except pd.errors.EmptyDataError:
        return None, "Error: The uploaded file has no data (EmptyDataError)."
    except pd.errors.ParserError as e:
        return None, f"Error parsing file: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error loading dataset: {str(e)}"


def get_dataset_overview(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Extracts an initial summary of the dataset for display upon file upload.

    Parameters:
        df (pd.DataFrame): The loaded dataset.
        dataset_name (str): The filename or label of the dataset.

    Returns:
        dict: Summary containing row count, column count, column names, data types,
              first 5 rows, and missing value counts.
    """
    if df is None or df.empty:
        return {}

    total_rows, total_cols = df.shape
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Format data types into human-readable strings
    column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": list(df.columns),
        "column_dtypes": column_dtypes,
        "missing_values": missing_per_col,
        "total_missing_values": total_missing,
        "head_preview": df.head(5),
    }


# Standalone demonstration block for testing and college viva explanation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing data_loader.py with sample dataset...")
    print("=" * 60)

    sample_csv_path = os.path.join("datasets", "sample_data.csv")
    loaded_df, err = load_dataset(sample_csv_path)

    if err:
        print(f"Failed to load: {err}")
    else:
        overview = get_dataset_overview(loaded_df, dataset_name="sample_data.csv")
        print(f"Dataset Name: {overview['dataset_name']}")
        print(f"Total Rows: {overview['total_rows']}")
        print(f"Total Columns: {overview['total_columns']}")
        print(f"Columns: {overview['column_names']}")
        print("\nMissing values per column:")
        for col, count in overview['missing_values'].items():
            print(f"  - {col}: {count}")
        print("\nFirst 5 rows:")
        print(overview['head_preview'])
        print("\nSUCCESS: data_loader.py works correctly!")
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
    - Median (50%): Middle value when sorted
    - Max: Largest value
    - Skewness: Asymmetry of the distribution
    - Kurtosis: Tailedness of the distribution
    - Outliers (IQR): Count of values beyond 1.5 * IQR

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary table of numerical statistics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        return pd.DataFrame(columns=["Column", "Count", "Mean", "Std Dev", "Min", "Median", "Max", "Skewness", "Kurtosis", "Outliers (IQR)"])

    records = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # IQR Outliers
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        skew_val = round(float(series.skew()), 2) if len(series) > 2 else 0.0
        kurt_val = round(float(series.kurtosis()), 2) if len(series) > 3 else 0.0

        records.append({
            "Column": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 2),
            "Std Dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "Min": round(float(series.min()), 2),
            "Median": round(float(series.median()), 2),
            "Max": round(float(series.max()), 2),
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Outliers (IQR)": outlier_count,
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


def detect_identifier_columns(df: pd.DataFrame) -> list:
    """Detects potential primary key, ID, or high-uniqueness identifier columns."""
    if df is None or df.empty:
        return []
    id_cols = []
    total = len(df)
    for col in df.columns:
        col_lower = str(col).lower()
        unique_cnt = df[col].nunique(dropna=True)
        # Check by name or 100% uniqueness for non-empty tables
        if any(kw in col_lower for kw in ["id", "uuid", "ssn", "passport", "code", "key"]):
            id_cols.append(col)
        elif total > 1 and unique_cnt == total:
            id_cols.append(col)
    return list(dict.fromkeys(id_cols))


def detect_constant_columns(df: pd.DataFrame) -> list:
    """Detects columns with zero variance or single unique value."""
    if df is None or df.empty:
        return []
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def get_memory_usage(df: pd.DataFrame) -> str:
    """Returns total deep memory usage formatted in KB or MB."""
    if df is None or df.empty:
        return "0 KB"
    bytes_used = df.memory_usage(deep=True).sum()
    if bytes_used >= 1024 * 1024:
        return f"{bytes_used / (1024 * 1024):.2f} MB"
    return f"{bytes_used / 1024:.2f} KB"


def get_text_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes string length statistics for free-text columns."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    records = []
    for col in text_cols:
        lengths = df[col].dropna().astype(str).str.len()
        if len(lengths) > 0:
            records.append({
                "Column": col,
                "Avg Length": round(float(lengths.mean()), 1),
                "Min Length": int(lengths.min()),
                "Max Length": int(lengths.max()),
            })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["Column", "Avg Length", "Min Length", "Max Length"])


def get_date_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes date range and duration for datetime columns."""
    date_cols = df.select_dtypes(include=["datetime"]).columns
    records = []
    for col in date_cols:
        series = df[col].dropna()
        if not series.empty:
            min_date = series.min()
            max_date = series.max()
            records.append({
                "Column": col,
                "Start Date": str(min_date),
                "End Date": str(max_date),
                "Days Range": (max_date - min_date).days if hasattr(max_date - min_date, "days") else "N/A",
            })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["Column", "Start Date", "End Date", "Days Range"])


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
    id_cols = detect_identifier_columns(df)
    const_cols = detect_constant_columns(df)
    mem_usage = get_memory_usage(df)
    text_summary = get_text_length_summary(df)
    date_summary = get_date_range_summary(df)

    total_cells = df.size
    total_missing_cells = int(df.isnull().sum().sum())
    overall_missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # Dataset Quality Score (0-100)
    completeness_score = max(0.0, 100.0 - overall_missing_pct)
    uniqueness_score = max(0.0, 100.0 - duplicate_info.get("duplicate_percentage", 0.0))
    quality_score = round(0.7 * completeness_score + 0.3 * uniqueness_score, 1)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": total_cells,
        "column_types": col_types,
        "duplicate_info": duplicate_info,
        "total_missing_cells": total_missing_cells,
        "overall_missing_percentage": overall_missing_pct,
        "quality_score": quality_score,
        "memory_usage": mem_usage,
        "identifier_columns": id_cols,
        "constant_columns": const_cols,
        "text_length_summary": text_summary,
        "date_range_summary": date_summary,
        "missing_summary_table": missing_df,
        "numerical_stats_table": num_stats,
        "categorical_stats_table": cat_stats,
    }


# Standalone demonstration block
if __name__ == "__main__":
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
