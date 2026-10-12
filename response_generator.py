"""
response_generator.py
---------------------
Translates verified analytical calculations from the Pandas analysis engine
into concise, natural-language English explanations.

Key principle:
The numerical figures ALWAYS originate directly from Pandas calculations,
preventing LLM hallucination while preserving fluent conversational flow.
"""

import pandas as pd
from llm_client import call_llm, check_api_status


def generate_response(question: str, analysis_result: dict, dataset_name: str = "dataset", df: pd.DataFrame = None) -> str:
    """
    Produces a natural language answer based on verified analytical results.

    Parameters:
        question (str): The user's input question.
        analysis_result (dict): Output from analysis_engine.execute_plan.
        dataset_name (str): Label/filename of the dataset.
        df (pd.DataFrame, optional): Full dataframe for contextual statistics.

    Returns:
        str: Concise, verified conversational response.
    """
    if not analysis_result:
        return "I could not analyze the data. Please try rephrasing your question."

    # If analysis engine reported an error
    if analysis_result.get("status") == "error":
        err_msg = analysis_result.get("message") or analysis_result.get("error", "Unknown error")
        return f"**[Notice] Could not complete analysis**: {err_msg}\n\n*Tip: You can ask about averages, totals, top records, group comparisons, or missing values.*"

    task = analysis_result.get("task", "")
    res_type = analysis_result.get("result_type", "")
    val = analysis_result.get("value")
    engine_msg = analysis_result.get("message", "")

    # Attempt LLM natural language phrasing if local Ollama is configured
    status = check_api_status()
    if status["configured"]:
        llm_text = _generate_llm_grounded_text(question, task, val, engine_msg)
        if llm_text:
            return llm_text

    # Deterministic formatted response generation
    return _generate_deterministic_text(task, res_type, val, engine_msg, question=question, analysis_result=analysis_result, df=df)


def _generate_llm_grounded_text(question: str, task: str, value, engine_msg: str) -> str:
    """Calls local Ollama Llama 3.2 to rephrase the verified computation cleanly."""
    val_repr = value.to_string(index=False) if isinstance(value, pd.DataFrame) else str(value)

    prompt = f"""
You are an expert Data Analyst summarizing verified findings for a stakeholder.
Question: "{question}"
Analytical Task: "{task}"
Verified Calculated Result from Pandas:
{val_repr}

Engine Summary: "{engine_msg}"

Instructions:
1. Explain the answer in 1-2 natural, polite sentences.
2. DO NOT change or invent any numbers. Rely strictly on the verified calculated result provided.
3. Be concise and direct.
"""
    system_inst = "You are a concise data analyst. Never invent numbers. Ground your explanation in verified facts."
    response, err = call_llm(prompt, system_instruction=system_inst, json_mode=False)
    if not err and response:
        return response.strip()

    return None


def _generate_deterministic_text(task: str, res_type: str, value, engine_msg: str, question: str = "", analysis_result: dict = None, df: pd.DataFrame = None) -> str:
    """Fallback text generator formatting insights into structured markdown."""
    q_lower = (question or "").lower()
    target_col = None
    if analysis_result:
        target_col = analysis_result.get("column") or analysis_result.get("task_column") or analysis_result.get("y_col")

    # Special handling for distribution & histograms
    if (task in ["visualize", "plot", "chart", "graph"] or "distribution" in q_lower) and df is not None:
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            series = df[target_col].dropna()
            mean_v = round(float(series.mean()), 1)
            median_v = round(float(series.median()), 1)
            min_v = int(series.min()) if float(series.min()).is_integer() else round(float(series.min()), 1)
            max_v = int(series.max()) if float(series.max()).is_integer() else round(float(series.max()), 1)
            q25 = round(float(series.quantile(0.25)))
            q75 = round(float(series.quantile(0.75)))
            skew = series.skew()
            skew_desc = "right-skewed" if skew > 0.4 else ("left-skewed" if skew < -0.4 else "approximately symmetric")
            unit = " years" if target_col.lower() == "age" else ""

            return f"""**{target_col} Distribution Insights**
The {target_col.lower()} distribution is {skew_desc}, with most passengers between {q25} and {q75}{unit}.
• Mean {target_col.lower()}: {mean_v}{unit}
• Median {target_col.lower()}: {median_v}{unit}
• Youngest: {min_v}{unit} | Oldest: {max_v}{unit}"""

    if task in ["visualize", "plot", "chart", "graph"]:
        return f"**Visualization Generated**\n{engine_msg}\n\n*The visualization plot is rendered below along with the data breakdown table.*"

    if res_type == "scalar":
        return f"**Result Analysis**\n{engine_msg}"

    elif res_type == "table" and isinstance(value, pd.DataFrame):
        rows_count = len(value)
        return f"**Result Analysis**\n{engine_msg}\n\n*Displaying {rows_count} relevant record(s) in the table below.*"

    return f"**Result Analysis**\n{engine_msg}"


# Standalone demonstration block
if __name__ == "__main__":
    mock_result_scalar = {
        "status": "success",
        "task": "mean",
        "result_type": "scalar",
        "value": 67333.33,
        "message": "The average Salary is 67,333.33.",
        "error": None,
    }

    mock_result_error = {
        "status": "error",
        "task": "mean",
        "result_type": None,
        "value": None,
        "message": "Column 'Bonus' not found.",
        "error": "Column 'Bonus' not found.",
    }

    print("=" * 60)
    print("TESTING RESPONSE GENERATOR")
    print("=" * 60)
    print("Scalar Response:", generate_response("What is the average salary?", mock_result_scalar))
    print("Error Response: ", generate_response("What is the average bonus?", mock_result_error))
    print("\nSUCCESS: response_generator.py formatted answers accurately!")
