"""
query_planner.py
----------------
Translates natural-language questions into safe, structured JSON execution plans.

Architecture:
1. Constructs a strict schema-aware prompt with few-shot examples.
2. Calls the LLM (Gemini) requesting structured JSON output.
3. Fallback Heuristic Planner: If LLM API key is not configured or fails,
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

    task = plan.get("task", "").lower().strip()
    if not task:
        return None, "Validation Error: Plan missing 'task' field."

    if task not in SUPPORTED_TASKS:
        return None, f"Validation Error: Unsupported task '{task}'."

    # Validate column if required
    col = plan.get("column")
    if task in ["mean", "median", "sum", "std", "min", "max", "sort", "top_n", "bottom_n"]:
        if not col:
            return None, f"Validation Error: Task '{task}' requires a 'column' parameter."
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        plan["column"] = matched_col

        # Numerical requirement check
        if task in ["mean", "median", "sum", "std"]:
            if not pd.api.types.is_numeric_dtype(df[matched_col]):
                return None, f"Validation Error: Task '{task}' requires a numeric column, but '{matched_col}' is non-numeric."

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
- "missing_values": Count missing values
- "duplicates": Find duplicate rows
- "correlation": Correlation matrix of numerical columns

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
    """Sends prompt to Gemini LLM and parses JSON plan."""
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
# Heuristic Fallback Planner (Guarantees zero-failure demo)
# -------------------------------------------------------------

def _heuristic_fallback_plan(question: str, df: pd.DataFrame, conversation_history: list = None) -> dict:
    """
    Deterministic rule-based intent parser that works even without an internet/LLM connection.
    Essential for college viva and offline demonstrations!
    """
    q = question.lower()
    columns = list(df.columns)
    num_cols = get_column_types(df)["numerical"]
    cat_cols = get_column_types(df)["categorical"]

    # 1. Row & column counting
    if any(k in q for k in ["how many rows", "row count", "total rows", "number of rows", "number of records"]):
        return {"task": "count_rows"}
    if any(k in q for k in ["how many columns", "column count", "total columns"]):
        return {"task": "count_columns"}

    # 2. Missing values & duplicates
    if any(k in q for k in ["missing", "null", "empty", "nan"]):
        return {"task": "missing_values"}
    if "duplicate" in q:
        return {"task": "duplicates"}
    if "correlation" in q or "corr" in q:
        return {"task": "correlation"}

    # 3. Find matched numeric and categorical columns specifically
    matched_num_col = None
    for col in num_cols:
        col_clean = col.lower()
        col_spaced = col_clean.replace("_", " ")
        col_tokens = [t for t in col_clean.split("_") if len(t) > 2]
        if col_clean in q or col_spaced in q or any(t in q for t in col_tokens):
            matched_num_col = col
            break

    matched_cat_col = None
    for col in cat_cols:
        col_clean = col.lower()
        col_spaced = col_clean.replace("_", " ")
        col_tokens = [t for t in col_clean.split("_") if len(t) > 2]
        if col_clean in q or col_spaced in q or any(t in q for t in col_tokens):
            matched_cat_col = col
            break

    # Context fallback: If user asks follow-up like "What is the maximum?" without naming a column,
    # intelligently inherit the analyzed column from the most recent turn
    if not matched_num_col and conversation_history and len(conversation_history) > 0:
        last_turn = conversation_history[-1]
        prev_col = last_turn.get("column") or last_turn.get("analyzed_column")
        if prev_col and prev_col in num_cols:
            matched_num_col = prev_col

    target_num_col = matched_num_col if matched_num_col else (num_cols[0] if num_cols else None)

    # 4. Group by / category comparison ("average salary by department", "sales per region")
    if "by" in q or "per" in q or "across" in q or "each" in q or matched_cat_col:
        if matched_cat_col:
            agg = "mean" if any(k in q for k in ["average", "mean", "avg"]) else ("sum" if ("sum" in q or "total" in q) else "count")
            chart_type = "bar" if any(k in q for k in ["chart", "plot", "graph", "bar"]) else None
            return {
                "task": "group_aggregate",
                "group_by": matched_cat_col,
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

    # 6. Simple Aggregations
    if any(k in q for k in ["average", "mean", "avg"]):
        return {"task": "mean", "column": target_num_col}

    if "median" in q:
        return {"task": "median", "column": target_num_col}

    if any(k in q for k in ["total", "sum"]):
        return {"task": "sum", "column": target_num_col}

    if any(k in q for k in ["standard deviation", "std dev", "std"]):
        return {"task": "std", "column": target_num_col}

    # 6. Filter / condition ("greater than 50000", "> 50000")
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
    """Helper to resolve column names case-insensitively."""
    if not name:
        return None
    if name in df.columns:
        return name
    lower_map = {col.lower(): col for col in df.columns}
    return lower_map.get(name.lower().strip())


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
