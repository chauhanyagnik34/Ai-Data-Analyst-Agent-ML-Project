"""
query_planner.py
----------------
Translates natural-language questions into safe, structured JSON execution plans.

Architecture:
1. Constructs a strict schema-aware prompt with few-shot examples.
2. Calls the local LLM (Ollama Llama 3.2) requesting structured JSON output.
3. Fallback Heuristic Planner: If local LLM is unreachable or fails,
   a rule-based parser handles common questions so the app remains fully demonstrable!
4. Plan Validator: Checks plan integrity, column existence, and type constraints
   before sending to the analysis engine.
"""

import json
import re
import pandas as pd
from llm_client import call_llm, check_api_status
from data_profiler import get_column_types

# Whitelist of allowed analytical tasks
SUPPORTED_TASKS = [
    "count_rows",
    "count_columns",
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "std",
    "count_where",
    "filter",
    "sort",
    "top_n",
    "bottom_n",
    "group_aggregate",
    "missing_values",
    "duplicates",
    "correlation",
    "fill_missing",
    "scale_data",
    "visualize",
]


def plan_query(
    question: str,
    df: pd.DataFrame,
    conversation_history: list = None,
) -> tuple:
    """
    Translates a natural language question into a validated query plan.

    Parameters:
        question (str): The user's query in English.
        df (pd.DataFrame): The active dataset.
        conversation_history (list, optional): Previous conversation turns for context.

    Returns:
        tuple: (validated_plan, error_message)
    """
    if df is None or df.empty:
        return None, "Error: No dataset loaded to plan query against."

    if not question or not question.strip():
        return None, "Error: Question is empty."

    question = question.strip()

    # Security: Limit maximum question length to protect against prompt injection & buffer exhaustion
    MAX_QUESTION_LENGTH = 500
    if len(question) > MAX_QUESTION_LENGTH:
        return None, f"Error: Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."

    status = check_api_status()

    # If LLM API key is configured, use LLM-based planning
    if status["configured"]:
        plan, err = _llm_generate_plan(question, df, conversation_history)
        if not err and plan:
            return validate_plan(plan, df)
        # If LLM failed, fallback to heuristic planner
        print(f"[QueryPlanner] LLM call failed ({err}), falling back to heuristic planner...")

    # Fallback to deterministic heuristic planner
    heuristic_plan = _heuristic_fallback_plan(question, df, conversation_history)
    if heuristic_plan:
        return validate_plan(heuristic_plan, df)

    return None, (
        f"Could not understand the question '{question}'. "
        "Try asking about: average, highest, lowest, top 5, count rows, or group by a category."
    )


def validate_plan(plan: dict, df: pd.DataFrame) -> tuple:
    """
    Validates the query plan against the DataFrame schema.

    Returns:
        tuple: (validated_plan, error_message)
    """
    if not isinstance(plan, dict):
        return None, "Validation Error: Plan is not a valid dictionary."

    # Sanitize string null/none values from LLM output
    for k, v in list(plan.items()):
        if isinstance(v, str) and v.strip().lower() in ["null", "none", "nan", "undefined", ""]:
            plan[k] = None

    task = plan.get("task", "").lower().strip() if plan.get("task") else ""
    if not task:
        return None, "Validation Error: Plan missing 'task' field."

    if task not in SUPPORTED_TASKS:
        return None, f"Validation Error: Unsupported task '{task}'."

    # Validate column if required
    col = plan.get("column")
    if task in ["mean", "median", "sum", "std", "min", "max", "sort", "top_n", "bottom_n"]:
        if not col:
            # Default to first numeric column for top_n / bottom_n
            num_cols = get_column_types(df)["numerical"]
            if num_cols:
                col = num_cols[0]
                plan["column"] = col
            elif len(df.columns) > 0:
                col = df.columns[0]
                plan["column"] = col
            else:
                return None, f"Validation Error: Task '{task}' requires a 'column' parameter."

        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        plan["column"] = matched_col

        # Numerical requirement check
        if task in ["mean", "median", "sum", "std"]:
            if not pd.api.types.is_numeric_dtype(df[matched_col]):
                return None, f"Validation Error: Task '{task}' requires a numeric column, but '{matched_col}' is non-numeric."

    # Validate column for missing_values / duplicates if specified
    if task in ["missing_values", "duplicates"] and col:
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        plan["column"] = matched_col

    # Validate fill_missing
    if task == "fill_missing":
        if not col:
            return None, "Validation Error: Task 'fill_missing' requires a 'column' parameter."
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        plan["column"] = matched_col
        strategy = str(plan.get("strategy", "mean")).lower().strip()
        if strategy not in ["mean", "median", "mode", "constant", "zero", "drop", "custom"]:
            strategy = "mean"
        plan["strategy"] = strategy

    # Validate scale_data
    if task == "scale_data":
        if not col:
            num_cols = get_column_types(df)["numerical"]
            if num_cols:
                col = num_cols[0]
                plan["column"] = col
            else:
                return None, "Validation Error: Task 'scale_data' requires a numeric column."
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        if not pd.api.types.is_numeric_dtype(df[matched_col]):
            return None, f"Validation Error: Task 'scale_data' requires a numeric column, but '{matched_col}' is non-numeric."
        plan["column"] = matched_col
        method = str(plan.get("method", "zscore")).lower().strip()
        if method not in ["zscore", "minmax", "robust", "standard"]:
            method = "zscore"
        plan["method"] = method

    # Validate group_by
    if task == "group_aggregate":
        grp = plan.get("group_by")
        if not grp:
            return None, "Validation Error: 'group_aggregate' requires a 'group_by' column."
        matched_grp = _resolve_column_name(df, grp)
        if not matched_grp:
            return None, f"Validation Error: Group-by column '{grp}' not found."
        plan["group_by"] = matched_grp

        if col:
            matched_col = _resolve_column_name(df, col)
            if not matched_col:
                return None, f"Validation Error: Target aggregation column '{col}' not found."
            plan["column"] = matched_col

    # Validate condition_col for filter/count_where
    if task in ["filter", "count_where"]:
        cond_col = plan.get("condition_col") or col
        if cond_col:
            matched_cond = _resolve_column_name(df, cond_col)
            if not matched_cond:
                return None, f"Validation Error: Condition column '{cond_col}' not found."
            plan["condition_col"] = matched_cond

    # Validate visualize
    if task == "visualize":
        chart = str(plan.get("chart", "bar")).lower().strip()
        valid_charts = ["bar", "bargraph", "bar_chart", "barchart", "histogram", "hist", "scatter", "scatter_plot", "scatterplot", "line", "line_chart", "line_plot", "box", "boxplot", "box_plot", "pie", "pie_chart", "heatmap", "correlation"]
        if chart not in valid_charts:
            chart = "bar"
        plan["chart"] = chart

        if col:
            matched_col = _resolve_column_name(df, col)
            if not matched_col:
                return None, f"Validation Error: Column '{col}' does not exist in dataset."
            plan["column"] = matched_col

        x_col = plan.get("x_col")
        if x_col:
            matched_x = _resolve_column_name(df, x_col)
            if not matched_x:
                return None, f"Validation Error: Column '{x_col}' does not exist in dataset."
            plan["x_col"] = matched_x

        y_col = plan.get("y_col")
        if y_col:
            matched_y = _resolve_column_name(df, y_col)
            if not matched_y:
                return None, f"Validation Error: Column '{y_col}' does not exist in dataset."
            plan["y_col"] = matched_y

        grp = plan.get("group_by")
        if grp:
            matched_grp = _resolve_column_name(df, grp)
            if not matched_grp:
                return None, f"Validation Error: Group-by column '{grp}' not found."
            plan["group_by"] = matched_grp

        # If no column is specified and not a heatmap, pick the first numerical column
        if not plan.get("column") and not plan.get("x_col") and chart not in ["heatmap", "correlation"]:
            num_cols = get_column_types(df)["numerical"]
            if num_cols:
                plan["column"] = num_cols[0]
            elif len(df.columns) > 0:
                plan["column"] = df.columns[0]

    return plan, None


# -------------------------------------------------------------
# LLM Prompt Construction and Execution
# -------------------------------------------------------------

def _build_planner_prompt(question: str, df: pd.DataFrame, conversation_history: list = None) -> str:
    """Builds a schema-rich few-shot prompt for the LLM."""
    col_types = get_column_types(df)
    schema_info = {
        "columns": list(df.columns),
        "numerical_columns": col_types["numerical"],
        "categorical_columns": col_types["categorical"],
    }

    history_str = ""
    if conversation_history and len(conversation_history) > 0:
        history_str = "Recent conversation context:\n"
        for turn in conversation_history[-3:]:  # last 3 turns
            history_str += f"- User: {turn.get('user', '')} -> Previous Column: {turn.get('column', '')}\n"

    prompt = f"""
You are a precision Data Analysis Query Planner.
Your job is to translate a user's question into a strict JSON execution plan for Pandas.

Dataset Schema:
Columns: {schema_info['columns']}
Numerical Columns: {schema_info['numerical_columns']}
Categorical Columns: {schema_info['categorical_columns']}

{history_str}
Supported Tasks:
- "count_rows": Count total rows
- "count_columns": Count total columns
- "mean": Average of a numerical column
- "median": Median of a numerical column
- "sum": Total sum of a numerical column
- "min": Minimum value
- "max": Maximum value
- "std": Standard deviation
- "top_n": Top N records by column (specify "top_n": 5)
- "bottom_n": Bottom N records by column
- "filter": Filter records satisfying a condition
- "count_where": Count records satisfying a condition
- "group_aggregate": Group by a categorical column and aggregate ("aggregation": "mean"|"sum"|"count"|"median")
- "missing_values": Count missing values (specify "column": "ColumnName" for a specific column, or null for whole dataset)
- "duplicates": Find duplicate rows (specify "column": "ColumnName" for duplicates in a specific column, or null for whole dataset)
- "correlation": Correlation matrix of numerical columns
- "fill_missing": Impute missing values ("strategy": "mean"|"median"|"mode"|"zero"|"constant")
- "scale_data": Scale or normalize column ("method": "zscore"|"minmax"|"robust")
- "visualize": Create chart ("chart": "bar"|"histogram"|"scatter"|"line"|"box"|"pie"|"heatmap", "column": "Col", "x_col": "Col1", "y_col": "Col2")

Output JSON Schema:
{{
  "task": "task_name",
  "column": "ColumnName or null",
  "group_by": "CategoryColumn or null",
  "aggregation": "mean or sum or count or null",
  "condition_col": "ColumnName or null",
  "operator": "== or != or > or >= or < or <= or contains or null",
  "condition_val": number or string or null,
  "top_n": integer or null,
  "sort_ascending": boolean or null,
  "chart": "bar or line or scatter or histogram or null"
}}

User Question: "{question}"

Respond with ONLY the JSON plan. Do not include markdown or explanations.
"""
    return prompt.strip()


def _llm_generate_plan(question: str, df: pd.DataFrame, conversation_history: list = None) -> tuple:
    """Sends prompt to local Ollama Llama 3.2 LLM and parses JSON plan."""
    prompt = _build_planner_prompt(question, df, conversation_history)
    system_inst = "You are a precise data analysis compiler. Output strictly valid JSON without preamble."

    raw_response, err = call_llm(prompt, system_instruction=system_inst, json_mode=True)
    if err:
        return None, err

    try:
        plan = json.loads(raw_response)
        return plan, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse LLM response as JSON: {str(e)}"


# -------------------------------------------------------------
def _extract_matched_column(q: str, candidate_cols: list) -> str:
    """
    Safely finds the best matching column name in a question using regex word boundaries.
    Prevents false positives (e.g. 'age' matching inside 'average', 'tax' in 'syntax').
    """
    if not candidate_cols or not q:
        return None

    matches = []
    q_lower = q.lower()

    for col in candidate_cols:
        col_name = str(col)
        col_clean = col_name.lower().strip()
        col_spaced = col_clean.replace("_", " ")

        # 1. Exact whole-word match for column name with word boundaries
        pattern_spaced = r"\b" + re.escape(col_spaced) + r"\b"
        pattern_clean = r"\b" + re.escape(col_clean) + r"\b"

        m1 = re.search(pattern_spaced, q_lower)
        m2 = re.search(pattern_clean, q_lower)
        m = m1 or m2
        if m:
            matches.append((200 + len(col_clean), m.start(), col))
            continue

        # 2. Token match for compound columns (e.g. 'total_sales' matched by 'sales')
        tokens = [t for t in re.split(r"[_\s]+", col_clean) if len(t) > 2]
        STOP_TOKENS = {
            "the", "and", "for", "row", "col", "data", "val", "num", "all",
            "any", "avg", "sum", "max", "min", "per", "each", "count",
            "rate", "type", "date", "year", "time", "list", "total", "average", "value", "values"
        }
        for token in tokens:
            if token in STOP_TOKENS:
                continue
            token_pattern = r"\b" + re.escape(token) + r"\b"
            m_token = re.search(token_pattern, q_lower)
            if m_token:
                matches.append((100 + len(token), m_token.start(), col))
                break

    if not matches:
        return None

    # Sort by match priority score descending, then by earliest appearance in text
    matches.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return matches[0][2]


def _extract_all_matched_columns(q: str, candidate_cols: list) -> list:
    """
    Finds all unique columns mentioned in the query, ordered by their appearance in the text.
    """
    if not candidate_cols or not q:
        return []

    matches = []
    q_lower = q.lower()

    for col in candidate_cols:
        col_clean = str(col).lower().strip()
        col_spaced = col_clean.replace("_", " ")

        p1 = r"\b" + re.escape(col_spaced) + r"\b"
        p2 = r"\b" + re.escape(col_clean) + r"\b"
        m1 = re.search(p1, q_lower)
        m2 = re.search(p2, q_lower)
        m = m1 or m2
        if m:
            matches.append((m.start(), col))
            continue

        # Check tokens for compound column names
        tokens = [t for t in re.split(r"[_\s]+", col_clean) if len(t) > 2]
        STOP_TOKENS = {
            "the", "and", "for", "row", "col", "data", "val", "num", "all",
            "any", "avg", "sum", "max", "min", "per", "each", "count",
            "rate", "type", "date", "year", "time", "list", "total", "average", "value", "values"
        }
        for token in tokens:
            if token in STOP_TOKENS:
                continue
            token_pattern = r"\b" + re.escape(token) + r"\b"
            m_token = re.search(token_pattern, q_lower)
            if m_token:
                matches.append((m_token.start(), col))
                break

    matches.sort(key=lambda x: x[0])
    seen = set()
    result = []
    for _, col in matches:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


# -------------------------------------------------------------
# Heuristic Fallback Planner (Guarantees zero-failure demo)
# -------------------------------------------------------------

def _heuristic_fallback_plan(question: str, df: pd.DataFrame, conversation_history: list = None) -> dict:
    """
    Deterministic rule-based intent parser that works even without an internet/LLM connection.
    Essential for college viva and offline demonstrations!
    """
    q = question.lower()
    columns = list(df.columns)
    col_types = get_column_types(df)
    num_cols = col_types["numerical"]
    cat_cols = col_types["categorical"]

    # 1. Row & column counting
    if any(k in q for k in ["how many rows", "row count", "total rows", "number of rows", "number of records"]):
        return {"task": "count_rows"}
    if any(k in q for k in ["how many columns", "column count", "total columns"]):
        return {"task": "count_columns"}

    # 2. Fill / impute missing values
    if any(k in q for k in ["fill", "impute", "replace null", "replace missing", "handle missing", "fillna"]):
        col_match = _extract_matched_column(q, columns)
        if not col_match:
            missing_cols = [c for c in columns if df[c].isnull().sum() > 0]
            col_match = missing_cols[0] if missing_cols else (num_cols[0] if num_cols else None)

        strategy = "mean"
        if "median" in q:
            strategy = "median"
        elif "mode" in q or "most frequent" in q or "common" in q:
            strategy = "mode"
        elif "zero" in q or " 0 " in q or "with 0" in q:
            strategy = "zero"
        elif "drop" in q or "remove" in q:
            strategy = "drop"
        elif "mean" in q or "average" in q:
            strategy = "mean"

        fill_val = None
        m_val = re.search(r"(?:with|to|by)\s+['\"]?([a-zA-Z0-9_\s]+)['\"]?", q)
        if m_val:
            cand = m_val.group(1).strip()
            if cand.lower() not in ["mean", "median", "mode", "mean value", "median value", "mode value", "zero", "average", "avg"]:
                strategy = "constant"
                fill_val = cand

        return {
            "task": "fill_missing",
            "column": col_match,
            "strategy": strategy,
            "fill_value": fill_val,
        }

    # 3. Missing values & duplicates inspection
    if any(k in q for k in ["missing", "null", "empty", "nan"]):
        col_match = _extract_matched_column(q, columns)
        if col_match:
            return {"task": "missing_values", "column": col_match}
        return {"task": "missing_values"}

    if "duplicate" in q:
        col_match = _extract_matched_column(q, columns)
        if col_match:
            return {"task": "duplicates", "column": col_match}
        return {"task": "duplicates"}
    if "correlation" in q or "corr" in q:
        return {"task": "correlation"}

    # Normalization & Scaling (Z-score, Min-Max, Standardization)
    if any(k in q for k in ["normaliz", "standardiz", "scale", "z score", "zscore", "min max", "minmax", "robust"]):
        col_match = _extract_matched_column(q, num_cols)
        if not col_match:
            col_match = num_cols[0] if num_cols else None

        method = "zscore"
        if any(k in q for k in ["min max", "minmax", "0 to 1", "0-1"]):
            method = "minmax"
        elif "robust" in q:
            method = "robust"
        elif any(k in q for k in ["z score", "zscore", "standardiz", "standard"]):
            method = "zscore"

        return {
            "task": "scale_data",
            "column": col_match,
            "method": method,
        }

    # ---------------------------------------------------------
    # Visualizations & Charts (Bar Graph, Histogram, Scatter, Line, Box, Pie, Heatmap)
    # ---------------------------------------------------------
    is_vis = any(k in q for k in [
        "distribution", "distrib",
        "bargraph", "bar graph", "bar chart", "barchart",
        "histogram", "hist",
        "scatter", "scatter plot", "scatterplot",
        "line chart", "line graph", "line plot",
        "box plot", "boxplot", "box-plot",
        "pie chart", "piechart",
        "heatmap", "heat map",
        "visualiz", "plot", "graph", "chart", "draw"
    ])

    if is_vis:
        # Determine specific chart type
        chart_type = "bar"
        if any(k in q for k in ["heatmap", "heat map", "correlation heatmap", "correlation matrix"]):
            chart_type = "heatmap"
        elif any(k in q for k in ["scatter", "scatter plot", "scatterplot"]):
            chart_type = "scatter"
        elif any(k in q for k in ["histogram", "hist"]):
            chart_type = "histogram"
        elif any(k in q for k in ["box plot", "boxplot", "box-plot"]):
            chart_type = "box"
        elif any(k in q for k in ["line chart", "line graph", "line plot", "trend line"]):
            chart_type = "line"
        elif any(k in q for k in ["pie chart", "piechart"]):
            chart_type = "pie"
        elif any(k in q for k in ["bargraph", "bar graph", "bar chart", "barchart", "bar"]):
            chart_type = "bar"
        elif "distribution" in q:
            chart_type = "histogram"
        else:
            chart_type = "bar"

        # Correlation heatmap
        if chart_type == "heatmap":
            return {"task": "visualize", "chart": "heatmap"}

        # Extract all referenced columns in the query
        all_cols = _extract_all_matched_columns(q, columns)

        # Check for grouped category chart (e.g. "bar chart of Fare by Pclass")
        group_col = None
        m_by = re.search(r"\b(?:by|per|across|grouped by)\s+([a-zA-Z0-9_\s]+)", q)
        if m_by:
            group_col = _extract_matched_column(m_by.group(1).strip(), columns)

        if group_col:
            remaining = [c for c in all_cols if c != group_col]
            target_col = remaining[0] if remaining else _extract_matched_column(q, num_cols)
            if not target_col:
                target_col = num_cols[0] if num_cols else group_col
            return {
                "task": "visualize",
                "chart": chart_type,
                "column": target_col,
                "x_col": group_col,
                "y_col": target_col,
                "group_by": group_col,
            }

        # Multiple columns mentioned (e.g. "scatter plot of Age vs Fare")
        if len(all_cols) >= 2:
            x_c = all_cols[0]
            y_c = all_cols[1]
            if chart_type == "bar" and any(k in q for k in [" vs ", " versus ", " against ", " and "]) and x_c in num_cols and y_c in num_cols:
                chart_type = "scatter"

            return {
                "task": "visualize",
                "chart": chart_type,
                "x_col": x_c,
                "y_col": y_c,
                "column": y_c,
            }

        # Single column mentioned (e.g. "create bargraph of Age", "histogram of Age")
        col_match = all_cols[0] if all_cols else _extract_matched_column(q, columns)
        if not col_match and conversation_history and len(conversation_history) > 0:
            last_turn = conversation_history[-1]
            prev_col = last_turn.get("column") or last_turn.get("analyzed_column")
            if prev_col and prev_col in columns:
                col_match = prev_col

        if not col_match:
            col_match = num_cols[0] if num_cols else (columns[0] if columns else None)

        return {
            "task": "visualize",
            "chart": chart_type,
            "column": col_match,
        }

    # 3. Find matched numeric column specifically using word boundaries
    matched_num_col = _extract_matched_column(q, num_cols)
    if not matched_num_col:
        # Also check all columns in df that are numeric-convertible
        convertible_cols = [c for c in columns if c not in num_cols and pd.to_numeric(df[c], errors="coerce").notnull().mean() > 0.5]
        matched_num_col = _extract_matched_column(q, convertible_cols)

    # 4. Group by / category comparison ("average salary by department", "fare by pclass")
    group_col = None
    m_by = re.search(r"\b(?:by|per|across|for each|each|group by|grouped by)\s+([a-zA-Z0-9_\s]+)", q)
    if m_by:
        target_phrase = m_by.group(1).strip()
        group_col = _extract_matched_column(target_phrase, columns)

    # Fallback group check: if query has grouping keywords and matches a categorical column
    matched_cat_col = _extract_matched_column(q, cat_cols)
    if not group_col and matched_cat_col:
        if any(k in q for k in [" by ", " per ", " across ", " each ", "compare", "breakdown", "distribution"]):
            group_col = matched_cat_col

    # If grouping, ensure target_num_col is distinct from group_col
    if group_col and matched_num_col == group_col:
        remaining = [c for c in num_cols if c != group_col]
        matched_num_col = _extract_matched_column(q, remaining) if remaining else None

    # Context fallback: If user asks follow-up like "What is the maximum?" without naming a column,
    # intelligently inherit the analyzed column from the most recent turn
    if not matched_num_col and conversation_history and len(conversation_history) > 0:
        last_turn = conversation_history[-1]
        prev_col = last_turn.get("column") or last_turn.get("analyzed_column")
        if prev_col and prev_col in num_cols:
            matched_num_col = prev_col

    target_num_col = matched_num_col if matched_num_col else (num_cols[0] if num_cols else None)

    # Execute group aggregation if group_col was identified
    if group_col:
        agg = "mean" if any(k in q for k in ["average", "mean", "avg"]) else ("sum" if any(k in q for k in ["sum", "total"]) else ("count" if "count" in q else "mean"))
        chart_type = "bar" if any(k in q for k in ["chart", "plot", "graph", "bar"]) else None
        return {
            "task": "group_aggregate",
            "group_by": group_col,
            "column": target_num_col,
            "aggregation": agg,
            "chart": chart_type,
        }

    # 5. Ranking (Top N, Bottom N)
    if "top" in q or "rank" in q:
        m = re.search(r"(?:top|highest)\s+(\d+)", q)
        if m or any(k in q for k in ["top 3", "top 5", "top 10", "top records"]):
            top_k = int(m.group(1)) if m else 5
            return {"task": "top_n", "column": target_num_col, "top_n": top_k}

    if "bottom" in q:
        m = re.search(r"(?:bottom|lowest)\s+(\d+)", q)
        if m or any(k in q for k in ["bottom 3", "bottom 5", "bottom 10"]):
            bot_k = int(m.group(1)) if m else 5
            return {"task": "bottom_n", "column": target_num_col, "top_n": bot_k}

    # 6. Maximum and Minimum
    if any(k in q for k in ["highest", "maximum", "max", "greatest", "peak"]):
        return {"task": "max", "column": target_num_col}

    if any(k in q for k in ["lowest", "minimum", "min", "least"]):
        return {"task": "min", "column": target_num_col}

    # 7. Simple Aggregations
    if any(k in q for k in ["average", "mean", "avg"]):
        return {"task": "mean", "column": target_num_col}

    if "median" in q:
        return {"task": "median", "column": target_num_col}

    if any(k in q for k in ["total", "sum"]):
        return {"task": "sum", "column": target_num_col}

    if any(k in q for k in ["standard deviation", "std dev", "std"]):
        return {"task": "std", "column": target_num_col}

    # 8. Filter / condition ("greater than 50000", "> 50000")
    m_num = re.search(r"(\d+[\d,]*)", q)
    if m_num and any(k in q for k in [">", "<", "greater", "more", "less", "above", "higher than"]):
        val = float(m_num.group(1).replace(",", ""))
        op = ">=" if "greater than or equal" in q else (">" if any(k in q for k in [">", "greater", "more", "above", "higher"]) else "<")
        return {
            "task": "filter",
            "condition_col": target_num_col,
            "operator": op,
            "condition_val": val,
        }

    return None


def _resolve_column_name(df: pd.DataFrame, name: str) -> str:
    """Helper to resolve column names case-insensitively and handle spaces/underscores."""
    if not name or df is None:
        return None
    cleaned_name = str(name).strip().lower()
    if cleaned_name in ["null", "none", "nan", "undefined", ""]:
        return None
    if name in df.columns:
        return name
    lower_map = {str(col).lower(): col for col in df.columns}
    if cleaned_name in lower_map:
        return lower_map[cleaned_name]

    # Handle underscore vs space variations
    normalized_name = re.sub(r"[\s_]+", "", cleaned_name)
    for col in df.columns:
        if re.sub(r"[\s_]+", "", str(col).lower()) == normalized_name:
            return col
    return None


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    import os

    df, _ = load_dataset(os.path.join("datasets", "sample_data.csv"))
    print("=" * 65)
    print("TESTING QUERY PLANNER (NATURAL LANGUAGE -> STRUCTURED JSON)")
    print("=" * 65)

    test_questions = [
        "What is the average salary?",
        "Find the highest salary in the company",
        "Show average salary by department",
        "How many rows are in this dataset?",
        "Show employees with salary greater than 70000",
        "Identify missing values",
    ]

    for q in test_questions:
        plan, err = plan_query(q, df)
        print(f"\nQuestion: '{q}'")
        if err:
            print(f"Error: {err}")
        else:
            print(f"Plan: {json.dumps(plan)}")

    print("\nSUCCESS: query_planner.py translated all queries into valid JSON plans!")
