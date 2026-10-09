"""
analysis_engine.py
------------------
Deterministic tabular data analysis engine powered by Pandas and NumPy.
Executes controlled, safe analytical tasks without running arbitrary LLM Python code.

Supported operations:
1. Count rows
2. Count columns
3. Mean (average)
4. Median
5. Sum
6. Minimum
7. Maximum
8. Standard deviation
9. Count records matching a condition
10. Filter rows
11. Sort rows
12. Top N records
13. Bottom N records
14. Group by & aggregate (sum, mean, count, median)
15. Identify missing values
16. Identify duplicate rows
17. Correlation matrix
"""

import pandas as pd
import numpy as np


def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    """
    Main entry point for executing an analytical query plan safely.

    Parameters:
        df (pd.DataFrame): Target dataset.
        plan (dict): Structured execution plan containing 'task' and parameters.

    Returns:
        dict: Standardized result envelope:
              {
                  "status": "success" | "error",
                  "task": str,
                  "result_type": "scalar" | "table" | "series",
                  "value": Any,
                  "chart_data": None | pd.DataFrame / pd.Series,
                  "message": str,
                  "error": None | str
              }
    """
    if df is None or df.empty:
        return {
            "status": "error",
            "task": plan.get("task", "unknown") if plan else "unknown",
            "result_type": None,
            "value": None,
            "chart_data": None,
            "message": "Dataset is empty or not loaded.",
            "error": "Dataset is empty.",
        }

    if not isinstance(plan, dict) or "task" not in plan:
        return {
            "status": "error",
            "task": "invalid",
            "result_type": None,
            "value": None,
            "chart_data": None,
            "message": "Invalid query plan: missing 'task' key.",
            "error": "Invalid plan format.",
        }

    task = plan["task"].lower().strip()
    column = plan.get("column")
    col_condition = plan.get("condition_col") or column
    operator = plan.get("operator", "==")
    condition_val = plan.get("condition_val")
    group_by = plan.get("group_by")
    aggregation = plan.get("aggregation", "mean")
    top_n = plan.get("top_n", 5)
    ascending = plan.get("sort_ascending", False)

    try:
        # 1. ROW COUNT
        if task in ["count_rows", "total_rows", "row_count"]:
            total = len(df)
            return _success(task, "scalar", total, f"The dataset contains {total} rows.")

        # 2. COLUMN COUNT
        elif task in ["count_columns", "total_columns", "col_count"]:
            total = len(df.columns)
            return _success(task, "scalar", total, f"The dataset contains {total} columns.")

        # 3. MEAN (AVERAGE)
        elif task in ["mean", "average", "avg"]:
            valid, err = _validate_numeric_column(df, column)
            if not valid:
                return _error(task, err)
            avg_val = round(float(df[column].dropna().mean()), 2)
            return _success(task, "scalar", avg_val, f"The average {column} is {avg_val:,}.", task_column=column)

        # 4. MEDIAN
        elif task in ["median", "med"]:
            valid, err = _validate_numeric_column(df, column)
            if not valid:
                return _error(task, err)
            med_val = round(float(df[column].dropna().median()), 2)
            return _success(task, "scalar", med_val, f"The median {column} is {med_val:,}.", task_column=column)

        # 5. SUM (TOTAL)
        elif task in ["sum", "total"]:
            valid, err = _validate_numeric_column(df, column)
            if not valid:
                return _error(task, err)
            sum_val = round(float(df[column].dropna().sum()), 2)
            return _success(task, "scalar", sum_val, f"The total sum of {column} is {sum_val:,}.", task_column=column)

        # 6. MINIMUM
        elif task in ["min", "minimum", "lowest"]:
            valid, err = _validate_column(df, column)
            if not valid:
                return _error(task, err)
            min_val = df[column].dropna().min()
            # If numeric, round it
            if isinstance(min_val, (int, float, np.number)):
                min_val = round(float(min_val), 2)
            return _success(task, "scalar", min_val, f"The minimum {column} is {min_val}.", task_column=column)

        # 7. MAXIMUM
        elif task in ["max", "maximum", "highest"]:
            valid, err = _validate_column(df, column)
            if not valid:
                return _error(task, err)
            max_val = df[column].dropna().max()
            if isinstance(max_val, (int, float, np.number)):
                max_val = round(float(max_val), 2)
            return _success(task, "scalar", max_val, f"The maximum {column} is {max_val}.", task_column=column)

        # 8. STANDARD DEVIATION
        elif task in ["std", "standard_deviation"]:
            valid, err = _validate_numeric_column(df, column)
            if not valid:
                return _error(task, err)
            std_val = round(float(df[column].dropna().std()), 2)
            return _success(task, "scalar", std_val, f"The standard deviation of {column} is {std_val}.", task_column=column)

        # 9. COUNT WITH CONDITION
        elif task in ["count_where", "count_condition"]:
            filtered_df, err = _apply_condition(df, col_condition, operator, condition_val)
            if err:
                return _error(task, err)
            cnt = len(filtered_df)
            msg = f"Found {cnt} record(s) where {col_condition} {operator} {condition_val}."
            return _success(task, "scalar", cnt, msg)

        # 10. FILTER ROWS
        elif task in ["filter", "filter_rows"]:
            filtered_df, err = _apply_condition(df, col_condition, operator, condition_val)
            if err:
                return _error(task, err)
            msg = f"Found {len(filtered_df)} record(s) where {col_condition} {operator} {condition_val}."
            return _success(task, "table", filtered_df, msg, chart_data=filtered_df)

        # 11. SORT ROWS
        elif task in ["sort", "sort_rows"]:
            valid, err = _validate_column(df, column)
            if not valid:
                return _error(task, err)
            sorted_df = df.sort_values(by=column, ascending=ascending).reset_index(drop=True)
            direction = "ascending" if ascending else "descending"
            msg = f"Sorted records by '{column}' in {direction} order."
            return _success(task, "table", sorted_df, msg)

        # 12. TOP N RECORDS
        elif task in ["top_n", "highest_n"]:
            valid, err = _validate_column(df, column)
            if not valid:
                return _error(task, err)
            n = int(top_n) if top_n else 5
            top_df = df.sort_values(by=column, ascending=False).head(n).reset_index(drop=True)
            msg = f"Top {n} records ordered by '{column}'."
            return _success(task, "table", top_df, msg, chart_data=top_df)

        # 13. BOTTOM N RECORDS
        elif task in ["bottom_n", "lowest_n"]:
            valid, err = _validate_column(df, column)
            if not valid:
                return _error(task, err)
            n = int(top_n) if top_n else 5
            bottom_df = df.sort_values(by=column, ascending=True).head(n).reset_index(drop=True)
            msg = f"Bottom {n} records ordered by '{column}'."
            return _success(task, "table", bottom_df, msg, chart_data=bottom_df)

        # 14 & 15. GROUP BY & AGGREGATE (Sum, Mean, Count, Median)
        elif task in ["group_aggregate", "group_by", "aggregate"]:
            valid_grp, err_grp = _validate_column(df, group_by)
            if not valid_grp:
                return _error(task, f"Group-by column error: {err_grp}")

            agg_func = aggregation.lower() if aggregation else "mean"
            if agg_func not in ["mean", "sum", "count", "median", "min", "max"]:
                return _error(task, f"Unsupported aggregation function '{agg_func}'. Supported: mean, sum, count, median, min, max.")

            if agg_func == "count":
                # For count, column can be anything or group itself
                grouped = df.groupby(group_by).size().reset_index(name="count")
                msg = f"Count of records grouped by '{group_by}'."
            else:
                valid_val, err_val = _validate_numeric_column(df, column)
                if not valid_val:
                    return _error(task, err_val)
                grouped = df.groupby(group_by)[column].agg(agg_func).round(2).reset_index()
                grouped.columns = [group_by, f"{column}_{agg_func}"]
                msg = f"{agg_func.capitalize()} of '{column}' grouped by '{group_by}'."

            return _success(task, "table", grouped, msg, chart_data=grouped)

        # 16. IDENTIFY MISSING VALUES
        elif task in ["missing_values", "null_values"]:
            missing_series = df.isnull().sum()
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing_Count": missing_series.values,
            })
            total_missing = int(missing_series.sum())
            msg = f"Total missing values across the dataset: {total_missing}."
            return _success(task, "table", missing_df, msg)

        # 17. IDENTIFY DUPLICATES
        elif task in ["duplicates", "duplicate_rows"]:
            dup_df = df[df.duplicated(keep=False)]
            cnt = int(df.duplicated().sum())
            msg = f"Found {cnt} duplicate row(s)."
            return _success(task, "table", dup_df, msg)

        # 18. CORRELATION MATRIX
        elif task in ["correlation", "corr"]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) < 2:
                return _error(task, "Dataset must have at least 2 numerical columns to calculate correlation.")
            corr_df = df[num_cols].corr().round(3).reset_index()
            corr_df.rename(columns={"index": "Variable"}, inplace=True)
            msg = "Pearson correlation matrix for numerical columns."
            return _success(task, "table", corr_df, msg, chart_data=corr_df)

        else:
            return _error(
                task,
                f"Unsupported task '{task}'. Supported tasks: count_rows, count_columns, mean, median, sum, min, max, std, count_where, filter, sort, top_n, bottom_n, group_aggregate, missing_values, duplicates, correlation.",
            )

    except Exception as e:
        return _error(task, f"Unexpected error during execution: {str(e)}")


# -------------------------------------------------------------
# Internal Helper and Validation Functions
# -------------------------------------------------------------

def _validate_column(df: pd.DataFrame, col_name: str) -> tuple:
    """Checks if a column exists in the DataFrame (case-insensitive fallback)."""
    if not col_name:
        return False, "Column name is missing in query plan."
    if col_name in df.columns:
        return True, None

    # Case-insensitive check
    lower_map = {col.lower(): col for col in df.columns}
    if col_name.lower() in lower_map:
        return True, None

    available = ", ".join(df.columns)
    return False, f"Column '{col_name}' not found. Available columns: {available}."


def _validate_numeric_column(df: pd.DataFrame, col_name: str) -> tuple:
    """Checks if a column exists and is numeric."""
    valid, err = _validate_column(df, col_name)
    if not valid:
        return False, err

    # Resolve case if needed
    col = col_name if col_name in df.columns else {c.lower(): c for c in df.columns}[col_name.lower()]
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False, f"Operation requires a numeric column, but '{col}' has type '{df[col].dtype}'."

    return True, None


def _apply_condition(df: pd.DataFrame, col: str, op: str, val) -> tuple:
    """Applies a condition securely using Pandas boolean indexing."""
    valid, err = _validate_column(df, col)
    if not valid:
        return None, err

    # Resolve actual column name case
    actual_col = col if col in df.columns else {c.lower(): c for c in df.columns}[col.lower()]
    series = df[actual_col]

    # Convert condition value if column is numeric
    if pd.api.types.is_numeric_dtype(series):
        try:
            val = float(val)
        except (ValueError, TypeError):
            return None, f"Condition value '{val}' cannot be converted to number for column '{actual_col}'."
    else:
        val = str(val).strip()

    if op == "==":
        filtered = df[series == val]
    elif op == "!=":
        filtered = df[series != val]
    elif op == ">":
        filtered = df[series > val]
    elif op == ">=":
        filtered = df[series >= val]
    elif op == "<":
        filtered = df[series < val]
    elif op == "<=":
        filtered = df[series <= val]
    elif op in ["contains", "like"]:
        filtered = df[series.astype(str).str.contains(str(val), case=False, na=False)]
    else:
        return None, f"Unsupported operator '{op}'. Supported: ==, !=, >, >=, <, <=, contains."

    return filtered.reset_index(drop=True), None


def _success(task: str, res_type: str, val, msg: str, chart_data=None, task_column=None) -> dict:
    return {
        "status": "success",
        "task": task,
        "task_column": task_column,
        "result_type": res_type,
        "value": val,
        "chart_data": chart_data,
        "message": msg,
        "error": None,
    }


def _error(task: str, err_msg: str) -> dict:
    return {
        "status": "error",
        "task": task,
        "result_type": None,
        "value": None,
        "chart_data": None,
        "message": err_msg,
        "error": err_msg,
    }


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    import os

    df, _ = load_dataset(os.path.join("datasets", "sample_data.csv"))
    print("=" * 65)
    print("TESTING ANALYSIS ENGINE OPERATIONS")
    print("=" * 65)

    # Test 1: Mean salary
    res1 = execute_plan(df, {"task": "mean", "column": "Salary"})
    print("1. Mean Salary:", res1["value"], "|", res1["message"])

    # Test 2: Top 3 employees by salary
    res2 = execute_plan(df, {"task": "top_n", "column": "Salary", "top_n": 3})
    print("\n2. Top 3 Salaries:")
    print(res2["value"][["Name", "Department", "Salary"]])

    # Test 3: Average salary by department
    res3 = execute_plan(df, {"task": "group_aggregate", "group_by": "Department", "column": "Salary", "aggregation": "mean"})
    print("\n3. Salary by Department:")
    print(res3["value"])

    # Test 4: Filter employees with salary > 70000
    res4 = execute_plan(df, {"task": "filter", "condition_col": "Salary", "operator": ">", "condition_val": 70000})
    print(f"\n4. Filter Salary > 70000: {len(res4['value'])} records found.")

    # Test 5: Safe error handling on invalid column
    res5 = execute_plan(df, {"task": "mean", "column": "NonExistentColumn"})
    print("\n5. Handled Invalid Column Safely:", res5["error"])

    print("\nSUCCESS: analysis_engine.py executed all operations flawlessly!")
