"""
Shared query workflow:

User question → Llama plan (heuristic fallback) → validated operation
→ Pandas/NumPy execution → grounded response → optional chart
"""

import re
from typing import Any

import numpy as np
import pandas as pd

from agent.executor import execute_plan, generate_response
from agent.ollama_client import call_llm, check_api_status
from agent.planner import plan_query
from core.visualization import create_auto_chart

CHART_TASKS = {
    "histogram",
    "box_plot",
    "scatter_plot",
    "pie_chart",
    "bar_chart",
    "line_chart",
    "count_plot",
    "heatmap",
    "correlation",
    "group_aggregate",
    "top_n",
    "bottom_n",
    "kmeans_cluster",
    "dbscan_cluster",
    "train_classification",
    "train_regression",
    "pca_reduction",
    "missing_values",
    "detect_outliers",
}

GREETINGS = {
    "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "sup", "yo", "who are you", "who are you?",
    "what can you do", "what can you do?", "help", "how to use", "how do i use this",
    "thanks", "thank you",
}


def is_conversational_query(question: str) -> bool:
    q_clean = (question or "").strip().lower()
    if q_clean in GREETINGS:
        return True
    words = re.findall(r"\b\w+\b", q_clean)
    if len(words) <= 3 and any(w in GREETINGS for w in words):
        analytical_keywords = {
            "average", "mean", "median", "sum", "count", "max", "min", "std", "chart",
            "plot", "histogram", "pie", "bar", "box", "scatter", "missing", "null",
            "filter", "predict", "train", "cluster", "drop", "fill",
        }
        if not any(w in words for w in analytical_keywords):
            return True
    return False


def generate_conversational_reply(question: str, df: pd.DataFrame, dataset_name: str) -> str:
    status = check_api_status()
    if status.get("configured"):
        cols_summary = ", ".join(df.columns[:20]) if df is not None and not df.empty else "None"
        num_cols = ", ".join(df.select_dtypes(include=[np.number]).columns[:5]) if df is not None and not df.empty else "None"
        cat_cols = ", ".join(df.select_dtypes(include=["object", "category"]).columns[:5]) if df is not None and not df.empty else "None"
        total_rows = len(df) if df is not None else 0
        total_cols = len(df.columns) if df is not None else 0
        model = status.get("model") or "Llama"

        sys_prompt = (
            "You are AI Agent, a helpful data analyst copilot powered by local Ollama.\n"
            f"Active dataset: '{dataset_name}' with {total_rows:,} rows and {total_cols} columns.\n"
            f"Columns: {cols_summary}.\n"
            f"Key numerical columns: {num_cols}. Key categorical columns: {cat_cols}.\n"
            f"Model in use: {model}.\n\n"
            "Greet the user, mention the loaded dataset, and suggest 3-4 concrete analysis questions.\n"
            "Do not invent statistics. Keep the reply concise markdown."
        )
        reply, err = call_llm(question, system_instruction=sys_prompt, timeout=45)
        if not err and reply and reply.strip():
            return reply.strip()
        if err:
            return (
                f"**Hello! I am your AI Data Analyst Agent.**\n\n"
                f"I loaded `{dataset_name}`, but the local Llama service is not ready:\n{err}\n\n"
                "You can still ask analysis questions. Pandas will compute the numbers."
            )

    col_sample = (
        df.columns[1] if df is not None and len(df.columns) > 1
        else (df.columns[0] if df is not None and len(df.columns) > 0 else "data")
    )
    return (
        f"**Hello! I am your AI Data Analyst Agent.**\n\n"
        f"I have loaded **`{dataset_name}`** with **{len(df):,} rows** and **{len(df.columns)} columns**.\n\n"
        f"{status.get('message', '')}\n\n"
        f"You can ask questions such as:\n"
        f"- *'What is the average {col_sample}?'*\n"
        f"- *'Create a pie chart for {col_sample}'*\n"
        f"- *'Show missing values across all columns'*\n"
        f"- *'Plot histogram for {col_sample}'*\n\n"
        f"What would you like to analyze?"
    )


def _summarize_value(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return value.head(8).to_string(index=False)
    if isinstance(value, pd.Series):
        return value.head(8).to_string()
    if isinstance(value, dict):
        items = list(value.items())[:8]
        return ", ".join(f"{k}={v}" for k, v in items)
    return str(value)


def _compare_previous(question: str, current_context: list, dataset_name: str) -> dict:
    if not current_context or len(current_context) < 1:
        return {
            "status": "error",
            "task": "compare_previous",
            "result_type": None,
            "value": None,
            "chart_data": None,
            "message": "There is no previous result to compare. Ask an analysis question first.",
            "error": "No previous result.",
        }

    last = current_context[-1]
    prior = current_context[-2] if len(current_context) >= 2 else None
    last_summary = last.get("result_summary") or last.get("message") or "No stored result."
    if prior:
        prior_summary = prior.get("result_summary") or prior.get("message") or "No stored result."
        message = (
            f"Previous question: {prior.get('user')}\n{prior_summary}\n\n"
            f"Latest question: {last.get('user')}\n{last_summary}\n\n"
            "These figures come from Pandas, not from the language model."
        )
        value = {
            "previous_question": prior.get("user"),
            "previous_result": prior_summary,
            "latest_question": last.get("user"),
            "latest_result": last_summary,
        }
    else:
        message = (
            f"Latest verified result for '{last.get('user')}':\n{last_summary}\n\n"
            "Ask another analysis question, then say 'compare this with the previous result'."
        )
        value = {"latest_question": last.get("user"), "latest_result": last_summary}

    return {
        "status": "success",
        "task": "compare_previous",
        "result_type": "scalar",
        "value": value,
        "chart_data": None,
        "message": message,
        "error": None,
    }


def handle_query(
    question: str,
    df: pd.DataFrame,
    dataset_name: str,
    context_history: list = None,
    query_df: pd.DataFrame = None,
) -> dict:
    """
    Run one user message through the planner and executor.

    Returns a dict with answer, plan, result, table-ready value, chart figure,
    updated dataframe, and updated conversation context.
    """
    current_context = list(context_history or [])
    dataset_name = dataset_name or "dataset"
    execution_df = query_df if query_df is not None else df

    if is_conversational_query(question):
        return {
            "answer": generate_conversational_reply(question, df, dataset_name),
            "plan": {"task": "conversational_reply"},
            "result": None,
            "figure": None,
            "updated_df": None,
            "context": current_context,
            "intent": None,
            "confidence": None,
        }

    plan, plan_err = plan_query(question, execution_df, conversation_history=current_context)
    if plan_err and query_df is not None and query_df is not df:
        alt_plan, alt_err = plan_query(question, df, conversation_history=current_context)
        if not alt_err:
            plan, plan_err = alt_plan, None
            execution_df = df

    if plan_err:
        if "did you mean" in plan_err.lower() or "not found" in plan_err.lower():
            answer = f"**[Notice]**: {plan_err}"
        else:
            answer = generate_conversational_reply(question, df, dataset_name)
            if plan_err:
                answer = f"**[Notice]**: {plan_err}\n\n{answer}"
        return {
            "answer": answer,
            "plan": {"task": "planning_error"},
            "result": None,
            "figure": None,
            "updated_df": None,
            "context": current_context,
            "intent": None,
            "confidence": None,
        }

    if plan.get("task") == "compare_previous":
        result = _compare_previous(question, current_context, dataset_name)
    else:
        result = execute_plan(execution_df, plan)
        if result.get("status") != "success" and query_df is not None and execution_df is not df:
            alt_res = execute_plan(df, plan)
            if alt_res.get("status") == "success":
                result = alt_res
                execution_df = df
        elif result.get("modified_df") is not None and query_df is not None and execution_df is not df:
            full_result = execute_plan(df, plan)
            if full_result.get("status") == "success":
                result = full_result
                execution_df = df

    target_column = plan.get("column") or plan.get("group_by") or plan.get("condition_col")
    current_context.append({
        "user": question,
        "task": plan.get("task"),
        "column": target_column,
        "analyzed_column": target_column,
        "message": result.get("message"),
        "result_summary": _summarize_value(result.get("value")),
    })
    if len(current_context) > 8:
        current_context = current_context[-8:]

    answer = generate_response(question, result, dataset_name=dataset_name, use_llm=False)

    figure = None
    if result.get("status") == "success" and plan.get("task") in CHART_TASKS:
        figure, _ = create_auto_chart(result, execution_df)

    return {
        "answer": answer,
        "plan": plan,
        "result": result,
        "figure": figure,
        "updated_df": result.get("modified_df") if result else None,
        "context": current_context,
        "intent": plan.get("task"),
        "confidence": None,
        "execution_df": execution_df,
    }
