"""
app.py
------
Main Gradio Web Application for AI Data Analyst Agent.
Integrates:
- File upload (CSV / Excel)
- Automated data profiling and schema summary
- Natural language query answering with Pandas
- Full KDD Data Preprocessing, Cleaning & Imputation (fill_missing, drop_missing, drop_duplicates, drop_column, normalize)
- Download Modified Data button right below the Ask button
- Dynamic visualization rendering (Matplotlib)
- Conversation history tracking
"""

import os
import re
import base64
import io
import json
import tempfile
import threading
import uuid
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response, FileResponse, RedirectResponse

from core.data_manager import load_dataset, get_dataset_overview, MAX_FILE_SIZE_BYTES, generate_full_profile, get_column_types
from agent.planner import plan_query
from agent.executor import execute_plan, generate_response
from core.visualization import create_auto_chart
from agent.ollama_client import check_api_status
from agent.pipeline import handle_query, is_conversational_query, generate_conversational_reply

# Ensure exports directory exists for modified dataset downloads
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# Global / Session State Helper
class SessionState:
    def __init__(self):
        self.df = None
        self.dataset_name = None
        self.conversation_history = []

    def reset(self):
        self.df = None
        self.dataset_name = None
        self.conversation_history = []


# -------------------------------------------------------------
# Gradio Event Handlers
# -------------------------------------------------------------

def on_file_upload(file_obj, current_history):
    """Handles dataset file upload, extracts profiling, and initializes preview."""
    if file_obj is None:
        return (
            None,  # preview_df
            "Please upload a CSV or Excel file to begin analysis.",  # status_markdown
            None,  # numerical_stats_df
            None,  # missing_summary_df
            [],    # chatbot reset
            None,  # result_table
            None,  # result_plot
            None,  # active_df
            "dataset",  # dataset_name_state
            None,  # btn_download
        )

    file_path = file_obj.name
    dataset_name = os.path.basename(file_path)

    df, err = load_dataset(file_path)
    if err:
        return (
            None,
            f"**Error Loading File**: {err}",
            None,
            None,
            [],
            None,
            None,
            None,
            dataset_name,
            None,
        )

    # Save initial export copy for downloading
    clean_name = os.path.splitext(dataset_name)[0].replace(" ", "_")
    export_file = os.path.join(EXPORTS_DIR, f"modified_{clean_name}.csv")
    df.to_csv(export_file, index=False)

    # Compute profiling summaries
    overview = get_dataset_overview(df, dataset_name=dataset_name)
    profile = generate_full_profile(df)

    status_text = (
        f"### Dataset: `{dataset_name}`\n"
        f"- **Rows**: {overview['total_rows']:,} | **Columns**: {overview['total_columns']}\n"
        f"- **Missing Cells**: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)\n"
        f"- **Duplicate Rows**: {profile['duplicate_info']['duplicate_count']}\n"
        f"- **Numerical Columns**: {', '.join(profile['column_types']['numerical']) or 'None'}\n"
        f"- **Categorical Columns**: {', '.join(profile['column_types']['categorical']) or 'None'}"
    )

    # Dynamic sample questions based on actual dataset columns
    num_cols = profile["column_types"]["numerical"]
    cat_cols = profile["column_types"]["categorical"]
    valid_num_cols = [c for c in num_cols if not c.lower().endswith("id") and c.lower() != "id"]
    chosen_num = valid_num_cols[0] if valid_num_cols else (num_cols[0] if num_cols else None)

    if "Salary" in df.columns:
        example_q1 = "'What is the average salary?'"
        example_q2 = "'Show average salary by department'" if "Department" in df.columns else "'Check for missing values'"
    elif chosen_num and cat_cols:
        example_q1 = f"'What is the average {chosen_num}?'"
        example_q2 = f"'Create pie chart for {cat_cols[0]}' or 'Show average {chosen_num} by {cat_cols[0]}'"
    elif chosen_num:
        example_q1 = f"'What is the average {chosen_num}?'"
        example_q2 = f"'Plot histogram for {chosen_num}'"
    else:
        example_q1 = "'How many rows are in this dataset?'"
        example_q2 = "'Check for missing values'"

    welcome_chat = [
        {
            "role": "assistant",
            "content": (
                f"Hello! I have loaded `{dataset_name}` with {overview['total_rows']:,} rows and {overview['total_columns']} columns. "
                f"You can ask questions in English like {example_q1} or {example_q2}."
            ),
        }
    ]

    return (
        df.head(10),                             # preview_df
        status_text,                             # status_markdown
        profile["numerical_stats_table"],        # numerical_stats_df
        profile["missing_summary_table"],        # missing_summary_df
        welcome_chat,                            # chatbot
        None,                                    # result_table
        None,                                    # result_plot
        df,                                      # active_df
        dataset_name,                            # dataset_name_state
        export_file,                             # btn_download
    )


def _is_conversational_query(question: str) -> bool:
    return is_conversational_query(question)


def _generate_conversational_reply(question: str, df: pd.DataFrame, dataset_name: str) -> str:
    return generate_conversational_reply(question, df, dataset_name)


def on_ask_question(question, active_df, dataset_name, chat_history, context_history):
    """Processes user questions via the LLM/Heuristic planner and analysis engine."""
    if not question or not question.strip():
        return (
            chat_history,
            context_history,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if active_df is None or (isinstance(active_df, pd.DataFrame) and active_df.empty):
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": "Please upload a CSV or Excel dataset first."})
        return (
            new_history,
            context_history,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    df = active_df
    dataset_name = dataset_name or "dataset"
    outcome = handle_query(question, df, dataset_name, context_history=list(context_history or []))
    current_context = outcome["context"]
    response_text = outcome["answer"]
    result = outcome.get("result") or {}
    chart_fig = outcome.get("figure")

    new_history = list(chat_history or [])
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": response_text})

    table_output = None
    if result.get("status") == "success" and isinstance(result.get("value"), pd.DataFrame):
        table_output = result["value"]

    if result.get("modified_df") is not None:
        updated_df = result["modified_df"]
        clean_name = os.path.splitext(dataset_name)[0].replace(" ", "_")
        export_file = os.path.join(EXPORTS_DIR, f"modified_{clean_name}.csv")
        updated_df.to_csv(export_file, index=False)

        overview = get_dataset_overview(updated_df, dataset_name=dataset_name)
        profile = generate_full_profile(updated_df)
        status_text = (
            f"### Dataset: `{dataset_name}` *(Modified)*\n"
            f"- **Rows**: {overview['total_rows']:,} | **Columns**: {overview['total_columns']}\n"
            f"- **Missing Cells**: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)\n"
            f"- **Duplicate Rows**: {profile['duplicate_info']['duplicate_count']}\n"
            f"- **Numerical Columns**: {', '.join(profile['column_types']['numerical']) or 'None'}\n"
            f"- **Categorical Columns**: {', '.join(profile['column_types']['categorical']) or 'None'}"
        )
        return (
            new_history,
            current_context,
            table_output,
            chart_fig,
            "",
            updated_df,
            export_file,
            updated_df.head(10),
            status_text,
            profile["numerical_stats_table"],
            profile["missing_summary_table"],
        )

    return (
        new_history,
        current_context,
        table_output,
        chart_fig,
        "",
        active_df,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def on_load_sample_dataset():
    """Loads the pre-packaged sample dataset for instant demonstration."""
    sample_path = os.path.join("datasets", "sample_data.csv")
    if not os.path.exists(sample_path):
        return None, "Sample dataset not found.", None, None, [], None, None, None, "dataset", None

    class DummyFile:
        def __init__(self, path):
            self.name = path

    dummy = DummyFile(sample_path)
    return on_file_upload(dummy, [])


def on_clear():
    """Resets interface state."""
    plt.close("all")
    return (
        None,  # file upload
        None,  # preview df
        "Upload a CSV or Excel file to begin analysis.",  # status markdown
        None,  # numerical stats
        None,  # missing summary
        [],    # chat history
        None,  # result table
        None,  # result plot
        "",    # input box
        None,  # active_df
        "dataset",  # dataset_name_state
        None,  # btn_download
    )


# -------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------

api_status = check_api_status()
api_status_badge = (
    f"🟢 **LLM**: Connected (Ollama - {api_status['model']})"
    if api_status["configured"]
    else f"🟡 **Mode**: Heuristic Fallback ({api_status['message']})"
)

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.app-header {
    text-align: center;
    padding: 12px;
    background: linear-gradient(135deg, #1E3A8A, #3B82F6);
    color: white;
    border-radius: 8px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="AI Data Analyst Agent") as demo:
    gr.HTML(
        """
        <div class="app-header">
            <h1 style="margin:0; font-size: 26px;">🤖 AI Data Analyst Agent</h1>
            <p style="margin:4px 0 0 0; font-size: 14px; opacity: 0.9;">
                Natural-Language Conversational Tabular Analysis Powered by Pandas, NumPy & local Ollama + Llama
            </p>
        </div>
        """
    )

    gr.Markdown(f"{api_status_badge} | **Supported Formats**: `.csv`, `.xlsx`, `.xls`")

    # Reactive states for session management
    active_df = gr.State(None)
    dataset_name_state = gr.State("dataset")
    context_history = gr.State([])

    with gr.Row():
        # LEFT COLUMN: File Upload & Profiling
        with gr.Column(scale=4):
            file_input = gr.File(
                label="📁 Upload Dataset (.csv or .xlsx)",
                file_types=[".csv", ".xlsx", ".xls"],
            )

            with gr.Row():
                btn_load_sample = gr.Button("📊 Load Sample Employee Dataset", variant="secondary")
                btn_clear = gr.Button("🔄 Reset Application", variant="stop")

            status_markdown = gr.Markdown("Upload a CSV or Excel file to begin analysis.")

            with gr.Accordion("📋 Dataset Profiling & Statistics", open=False):
                with gr.Tab("Data Preview"):
                    preview_df = gr.Dataframe(label="First 10 Rows", interactive=False)
                with gr.Tab("Numerical Statistics"):
                    numerical_stats_df = gr.Dataframe(label="Descriptive Statistics", interactive=False)
                with gr.Tab("Missing Values"):
                    missing_summary_df = gr.Dataframe(label="Missing Values Breakdown", interactive=False)

        # RIGHT COLUMN: Chat, Tables, and Charts
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="💬 Conversational Data Assistant",
                height=350,
            )

            # Query input and actions row (Ask button + Download button below)
            with gr.Row():
                query_input = gr.Textbox(
                    label="Ask a question about your data",
                    placeholder="e.g., Check missing values | Fill missing values in Age by mean | Correlation matrix",
                    scale=7,
                )
                with gr.Column(scale=3):
                    btn_send = gr.Button("Ask", variant="primary")
                    btn_download = gr.DownloadButton(
                        "📥 Download Modified Data",
                        variant="secondary",
                        size="sm",
                        visible=True,
                    )

            # Quick Question Suggestions - General Tasks & KDD Operations
            gr.Markdown("**🌐 Quick Prompts — General Tasks & KDD Operations:**")
            with gr.Row():
                btn_missing = gr.Button("🔍 Missing Values", size="sm")
                btn_corr = gr.Button("📈 Correlation Matrix", size="sm")
                btn_rows = gr.Button("🔢 Total Rows", size="sm")
                btn_cols = gr.Button("📊 Total Columns", size="sm")
                btn_dups = gr.Button("👥 Duplicate Rows", size="sm")
                btn_top = gr.Button("📋 Top 5 Records", size="sm")
                btn_outliers = gr.Button("⚠️ Detect Outliers", size="sm")
                btn_cluster = gr.Button("🎯 Cluster Data", size="sm")

            with gr.Accordion("💼 Sample Dataset Prompts (Employee Data)", open=False):
                with gr.Row():
                    q1 = gr.Button("Average Salary", size="sm")
                    q2 = gr.Button("Salary by Department", size="sm")
                    q3 = gr.Button("Top 3 Salaries", size="sm")
                    q4 = gr.Button("Salary > 70000", size="sm")

            with gr.Accordion("📊 Detailed Results & Charts", open=True):
                result_table = gr.Dataframe(label="Query Result Table", interactive=False)
                result_plot = gr.Plot(label="Generated Visualization")

    # -------------------------------------------------------------
    # UI Interactions & Event Wiring
    # -------------------------------------------------------------

    # File upload event
    file_input.change(
        fn=lambda f, c: on_file_upload(f, c) + ([],),
        inputs=[file_input, chatbot],
        outputs=[
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )

    # Load sample dataset button
    btn_load_sample.click(
        fn=lambda: on_load_sample_dataset() + ([],),
        inputs=[],
        outputs=[
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )

    # Question asking triggers
    ask_inputs = [query_input, active_df, dataset_name_state, chatbot, context_history]
    ask_outputs = [
        chatbot,
        context_history,
        result_table,
        result_plot,
        query_input,
        active_df,
        btn_download,
        preview_df,
        status_markdown,
        numerical_stats_df,
        missing_summary_df,
    ]

    btn_send.click(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)
    query_input.submit(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)

    # General task quick prompt buttons
    btn_missing.click(lambda: "Check for missing values in this dataset", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_corr.click(lambda: "Show correlation between numerical columns", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_rows.click(lambda: "How many rows are in this dataset?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_cols.click(lambda: "How many columns are in this dataset?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_dups.click(lambda: "Check for duplicate rows in this dataset", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_top.click(lambda: "Show the top 5 records", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_outliers.click(lambda: "Detect outliers in the numerical columns", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_cluster.click(lambda: "Cluster records into 3 groups using KMeans", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )

    # Sample employee quick prompt buttons
    q1.click(lambda: "What is the average salary?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q2.click(lambda: "Show average salary by department", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q3.click(lambda: "Top 3 employees by salary", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q4.click(lambda: "Filter employees with salary greater than 70000", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )

    # Clear button (resets all visual components and context state)
    btn_clear.click(
        fn=lambda: on_clear() + ([],),
        inputs=[],
        outputs=[
            file_input,
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            query_input,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )


# -------------------------------------------------------------
# Local API used by the dashboard and analysis tools
# -------------------------------------------------------------

API_DATASETS = {}
API_DATASETS_LOCK = threading.RLock()


def _frame_records(frame):
    """Convert a dataframe to JSON-safe records, including dates and nulls."""
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def _dataset_payload(dataset_id, record):
    df = record["df"]
    profile = generate_full_profile(df)
    types = get_column_types(df)
    numeric = set(types["numerical"])
    categorical = set(types["categorical"])
    dates = set(types["datetime"])
    columns = []
    for column in df.columns:
        if column in numeric:
            data_type = "Numeric"
        elif column in dates:
            data_type = "Date"
        elif column in categorical:
            data_type = "Categorical"
        else:
            data_type = "Other"
        columns.append({
            "name": str(column),
            "type": data_type,
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        })

    return {
        "id": dataset_id,
        "name": record["name"],
        "file_size": record["file_size"],
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "headers": [str(column) for column in df.columns],
        "columns": columns,
        "preview_rows": _frame_records(df.head(8)),
        "missing_count": int(profile["total_missing_cells"]),
        "missing_percentage": float(profile["overall_missing_percentage"]),
        "completeness": round(100.0 - float(profile["overall_missing_percentage"]), 2),
        "duplicate_count": int(profile["duplicate_info"]["duplicate_count"]),
        "quality_score": float(profile["quality_score"]),
        "memory_usage": profile["memory_usage"],
        "numeric_stats": _frame_records(profile["numerical_stats_table"]),
        "categorical_stats": _frame_records(profile["categorical_stats_table"]),
        "missing_summary": _frame_records(profile["missing_summary_table"]),
        "column_types": profile.get("column_types", {}),
        "identifier_columns": profile.get("identifier_columns", []),
        "constant_columns": profile.get("constant_columns", []),
        "about_data": {
            "name": record["name"],
            "summary": f"{record['name']} contains {len(df):,} records and {len(df.columns)} features. Overall data completeness is {round(100.0 - float(profile['overall_missing_percentage']), 1)}% with a data quality score of {profile['quality_score']}%.",
            "numerical_count": len(types["numerical"]),
            "categorical_count": len(types["categorical"]),
            "datetime_count": len(types["datetime"]),
        },
    }


def _store_api_dataset(df, filename, file_size):
    dataset_id = uuid.uuid4().hex
    record = {
        "df": df,
        "original_df": df.copy(deep=True),
        "name": filename,
        "file_size": int(file_size),
        "context_history": [],
    }
    with API_DATASETS_LOCK:
        API_DATASETS[dataset_id] = record
    return _dataset_payload(dataset_id, record)


web_app = FastAPI(title="AI Data Analyst Agent API")

@web_app.get("/api/health")
async def api_health():
    status = check_api_status()
    return {
        "status": "ok",
        "analytics": "connected",
        "llm_connected": bool(status.get("configured")),
        "llm_model": status.get("model"),
        "llm_message": status.get("message", ""),
    }


@web_app.post("/api/datasets")
async def api_upload_dataset(request: Request):
    filename = request.query_params.get("filename", "dataset.csv").replace("\\", "/").rsplit("/", 1)[-1]
    extension = os.path.splitext(filename)[1].lower()
    if extension not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Upload a CSV, XLSX, or XLS file.")
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller.")

    contents = await request.body()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        df, error = load_dataset(temp_path)
        if error:
            raise HTTPException(status_code=400, detail=error)
        return _store_api_dataset(df, filename, len(contents))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@web_app.post("/api/datasets/sample")
async def api_load_sample_dataset():
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Titanic-Dataset.csv")
    if not os.path.isfile(sample_path):
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "sample_data.csv")
    df, error = load_dataset(sample_path)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return _store_api_dataset(df, os.path.basename(sample_path), os.path.getsize(sample_path))


@web_app.get("/api/datasets/{dataset_id}")
async def api_get_dataset(dataset_id: str):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")
    return _dataset_payload(dataset_id, record)


@web_app.get("/api/datasets/{dataset_id}/rows")
async def api_get_dataset_rows(dataset_id: str, request: Request):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")

    try:
        offset = max(0, int(request.query_params.get("offset", "0")))
        limit = min(100, max(1, int(request.query_params.get("limit", "10"))))
    except ValueError:
        raise HTTPException(status_code=400, detail="Offset and limit must be whole numbers.")

    frame = record["df"]
    query = request.query_params.get("search", "").strip()[:120]
    filtered = frame
    if query:
        matches = frame.astype("string").apply(
            lambda column: column.str.contains(query, case=False, regex=False, na=False)
        ).any(axis=1)
        filtered = frame.loc[matches]

    page = filtered.iloc[offset:offset + limit]
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": _frame_records(page),
        "total": int(len(filtered)),
        "offset": offset,
        "limit": limit,
    }


@web_app.post("/api/datasets/{dataset_id}/reset")
async def api_reset_dataset(dataset_id: str):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")
        record["df"] = record["original_df"].copy(deep=True)
        record["context_history"].clear()
    return _dataset_payload(dataset_id, record)


@web_app.post("/api/datasets/{dataset_id}/ask")
async def api_ask_dataset(dataset_id: str, request: Request):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a JSON body with a question field.")
    question = str(body.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question about the dataset.")

    df = record["df"]
    dataset_name = record["name"]
    submitted_columns = body.get("columns", [])
    columns_by_name = {str(column): column for column in df.columns}
    selected_columns = [columns_by_name[name] for name in submitted_columns if isinstance(name, str) and name in columns_by_name]
    query_df = df.loc[:, selected_columns] if selected_columns and len(selected_columns) < len(df.columns) else df

    outcome = handle_query(
        question,
        df,
        dataset_name,
        context_history=record["context_history"],
        query_df=query_df,
    )
    record["context_history"] = outcome["context"]
    answer = outcome["answer"]
    plan = outcome.get("plan") or {"task": "conversational_reply"}
    result = outcome.get("result") or {}
    table = None
    chart_data = None
    updated_dataset = None
    intent = None
    confidence = None

    if plan.get("task") not in {None, "conversational_reply", "planning_error"}:
        from core.query_classifier import classify_user_query
        intent, confidence = classify_user_query(question)

    value = result.get("value")
    if isinstance(value, pd.Series):
        value = value.reset_index()
    if isinstance(value, pd.DataFrame):
        table = {"columns": [str(column) for column in value.columns], "rows": _frame_records(value)}
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        table = {"columns": list(value[0].keys()), "rows": value}
    elif isinstance(result.get("chart_data"), pd.DataFrame):
        cd = result["chart_data"]
        table = {"columns": [str(c) for c in cd.columns], "rows": _frame_records(cd)}

    figure = outcome.get("figure")
    if figure is not None:
        image_buffer = io.BytesIO()
        try:
            figure.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
        finally:
            plt.close(figure)
        chart_data = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")

    modified_df = outcome.get("updated_df")
    if modified_df is not None:
        record["df"] = modified_df
        updated_dataset = _dataset_payload(dataset_id, record)

    return {
        "answer": answer,
        "intent": str(intent) if intent is not None else None,
        "confidence": float(confidence) if confidence is not None else None,
        "plan": plan,
        "table": table,
        "chart": chart_data,
        "dataset": updated_dataset,
    }


@web_app.get("/api/datasets/{dataset_id}/export")
async def api_export_dataset(dataset_id: str):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(record["name"])[0]) or "dataset"
    csv_data = record["df"].to_csv(index=False).encode("utf-8-sig")
    return Response(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="modified_{safe_stem}.csv"'},
    )


# Helper for converting values safely to JSON
def _make_json_safe(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return _frame_records(obj.reset_index() if isinstance(obj, pd.Series) else obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(x) for x in obj]
    return obj


@web_app.get("/api/datasets/presets")
async def api_get_presets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    presets = []
    titanic_path = os.path.join(base_dir, "Titanic-Dataset.csv")
    if os.path.isfile(titanic_path):
        presets.append({
            "key": "titanic",
            "name": "Titanic Passenger Dataset",
            "filename": "Titanic-Dataset.csv",
            "description": "891 passenger records with survival outcomes, demographics, and fare tiers.",
            "records": 891,
            "columns": 12,
        })
    employee_path = os.path.join(base_dir, "datasets", "sample_data.csv")
    if os.path.isfile(employee_path):
        presets.append({
            "key": "employee",
            "name": "Corporate Employee Dataset",
            "filename": "sample_data.csv",
            "description": "15 employee profiles across engineering, sales, marketing with salary and ratings.",
            "records": 15,
            "columns": 9,
        })
    return {"presets": presets}


@web_app.post("/api/datasets/presets/{preset_key}")
async def api_load_preset(preset_key: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if preset_key.lower() in ["titanic", "titanic-dataset", "titanic_dataset"]:
        target_path = os.path.join(base_dir, "Titanic-Dataset.csv")
    elif preset_key.lower() in ["employee", "sample", "sample_data", "sample_data.csv"]:
        target_path = os.path.join(base_dir, "datasets", "sample_data.csv")
    else:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_key}' not found.")

    if not os.path.isfile(target_path):
        raise HTTPException(status_code=404, detail="Preset file does not exist on server.")

    df, error = load_dataset(target_path)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return _store_api_dataset(df, os.path.basename(target_path), os.path.getsize(target_path))


@web_app.post("/api/datasets/{dataset_id}/execute")
async def api_execute_dataset(dataset_id: str, request: Request):
    with API_DATASETS_LOCK:
        record = API_DATASETS.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the file again.")

    try:
        plan = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a valid JSON body with task parameters.")

    if not isinstance(plan, dict) or "task" not in plan:
        raise HTTPException(status_code=400, detail="Plan must contain a valid 'task' property.")

    df = record["df"]
    dataset_name = record["name"]
    context_history = record["context_history"]

    result = execute_plan(df, plan)
    table = None
    chart_data = None
    updated_dataset = None

    value = result.get("value")
    if isinstance(value, pd.Series):
        value = value.reset_index()
    if isinstance(value, pd.DataFrame):
        table = {"columns": [str(c) for c in value.columns], "rows": _frame_records(value)}
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        table = {"columns": list(value[0].keys()), "rows": value}
    elif isinstance(result.get("chart_data"), pd.DataFrame):
        cd = result["chart_data"]
        table = {"columns": [str(c) for c in cd.columns], "rows": _frame_records(cd)}

    if result.get("status") == "success":
        figure, _ = create_auto_chart(result, df)
        if figure is not None:
            image_buffer = io.BytesIO()
            try:
                figure.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
            finally:
                plt.close(figure)
            chart_data = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")

    # If dataset was modified, save to exports and update session
    modified_df = result.get("modified_df")
    if modified_df is not None:
        record["df"] = modified_df
        clean_name = os.path.splitext(dataset_name)[0].replace(" ", "_")
        export_file = os.path.join(EXPORTS_DIR, f"modified_{clean_name}.csv")
        modified_df.to_csv(export_file, index=False)
        updated_dataset = _dataset_payload(dataset_id, record)

    # Record turn context
    target_column = plan.get("column") or plan.get("group_by") or plan.get("condition_col")
    context_history.append({
        "user": f"Executed operation: {plan.get('task')}",
        "task": plan.get("task"),
        "column": target_column,
        "analyzed_column": target_column,
    })

    simulated_q = f"Operation: {plan.get('task')}"
    answer = generate_response(simulated_q, result, dataset_name=dataset_name)

    return {
        "status": result.get("status", "success"),
        "task": result.get("task"),
        "result_type": result.get("result_type"),
        "message": result.get("message"),
        "answer": answer,
        "value": _make_json_safe(result.get("value")),
        "table": table,
        "chart": chart_data,
        "metrics": _make_json_safe(result.get("metrics")),
        "dataset": updated_dataset,
    }


@web_app.post("/api/intent/classify")
async def api_intent_classify(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a valid JSON body with a 'query' field.")

    query = str(body.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="Provide a non-empty query to classify.")

    from core.query_classifier import get_trained_classifier, preprocess_text
    clf = get_trained_classifier()
    pred_label, confidence = clf.predict(query)
    cleaned_q = preprocess_text(query)
    probs = clf.pipeline.predict_proba([cleaned_q])[0]
    classes = list(clf.pipeline.classes_)

    distribution = [
        {"intent": c, "probability": round(float(p), 4), "percentage": round(float(p) * 100, 2)}
        for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    ]

    return {
        "query": query,
        "cleaned_query": cleaned_q,
        "predicted_intent": pred_label,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 1),
        "distribution": distribution,
    }


@web_app.get("/api/intent/metrics")
async def api_intent_metrics():
    from core.query_classifier import get_trained_classifier
    clf = get_trained_classifier()
    metrics = clf.train_and_evaluate(test_size=0.25, random_state=42)
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "accuracy_percentage": round(float(metrics["accuracy"]) * 100, 2),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics["weighted_f1"]), 4),
        "total_classes": int(metrics["total_classes"]),
        "classes": list(clf.pipeline.classes_),
        "train_samples": int(metrics["train_samples"]),
        "test_samples": int(metrics["test_samples"]),
    }


# Mount Modern Multi-Page Frontend Static Files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)
web_app.mount("/demo", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_demo")

@web_app.get("/")
async def root_app():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return RedirectResponse(url="/gradio")


@web_app.get("/agent", include_in_schema=False)
async def agent_app():
    return RedirectResponse(url="/#page-ai")


# Serve the existing Gradio application at /gradio so all workflows and tests remain accessible.
web_app = gr.mount_gradio_app(web_app, demo, path="/gradio")


if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Data Analyst Agent...")
    print("=" * 60)
    print("Dashboard:                   http://127.0.0.1:7860/")
    print("AI Agent:                    http://127.0.0.1:7860/agent")
    print("Legacy Gradio Dashboard:     http://127.0.0.1:7860/gradio")
    print("Interactive Swagger Docs:    http://127.0.0.1:7860/docs")
    uvicorn.run(web_app, host="127.0.0.1", port=int(os.environ.get("ADAA_PORT", "7860")))
