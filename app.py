"""
app.py
------
Modern, High-Fidelity UI for the AI Data Analyst Agent in Gradio.
Features:
- Sleek dark slate aesthetic matching publication UI
- Dataset metadata banner (file name, rows, columns, missing cells, duplicate status)
- Interactive drag-and-drop file ingestion
- Data preprocessing (Missing Value Imputation & Feature Scaling)
- Real-time conversational analytics assistant (Pandas deterministic computation)
- Quick prompt task triggers (Missing values, Correlation matrix, Distribution chart, etc.)
- Tabular and dynamic Matplotlib dark-theme visualizations
"""

import os
import tempfile
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Import core analytical modules
from data_loader import load_dataset, get_dataset_overview
from data_profiler import generate_full_profile
from query_planner import plan_query
from analysis_engine import execute_plan
from response_generator import generate_response
from visualization import create_auto_chart, create_histogram
from llm_client import check_api_status
from query_classifier import classify_user_query


# -------------------------------------------------------------
# HTML Template Builders
# -------------------------------------------------------------

def build_dataset_banner_html(dataset_name=None, overview=None, profile=None):
    """Renders the top horizontal dataset status bar."""
    if not dataset_name or not overview or not profile:
        return """
        <div class="dataset-meta-bar empty-meta-bar">
            <div class="meta-left">
                <span class="file-icon">📁</span>
                <span class="file-name-text">No dataset loaded</span>
            </div>
            <div class="meta-stats">
                <span class="meta-item"><span class="meta-dot dot-gray"></span> Upload a CSV or Excel file to begin analysis</span>
            </div>
        </div>
        """

    rows = overview.get("total_rows", 0)
    cols = overview.get("total_columns", 0)
    missing = profile.get("total_missing_cells", 0)
    duplicates = profile.get("duplicate_info", {}).get("duplicate_count", 0)

    return f"""
    <div class="dataset-meta-bar">
        <div class="meta-left">
            <span class="file-sheet-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="8" y1="13" x2="16" y2="13"></line>
                    <line x1="8" y1="17" x2="16" y2="17"></line>
                </svg>
            </span>
            <span class="file-name-text">{dataset_name}</span>
        </div>
        <div class="meta-stats">
            <div class="meta-item"><span class="meta-dot dot-green"></span> Dataset loaded</div>
            <div class="meta-divider"></div>
            <div class="meta-item"><span class="meta-icon">📑</span> {rows:,} rows</div>
            <div class="meta-divider"></div>
            <div class="meta-item"><span class="meta-icon">◫</span> {cols} columns</div>
            <div class="meta-divider"></div>
            <div class="meta-item warning-pill"><span class="meta-icon">⚠️</span> {missing:,} missing cells</div>
            <div class="meta-divider"></div>
            <div class="meta-item"><span class="meta-dot dot-green"></span> {duplicates} duplicates</div>
        </div>
    </div>
    """


def build_file_chip_html(dataset_name=None, file_size_kb=None):
    """Renders the loaded file chip below the upload dropzone."""
    if not dataset_name:
        return ""
    size_str = f"({file_size_kb:.1f} KB)" if file_size_kb else ""
    return f"""
    <div class="active-file-chip">
        <div class="chip-info">
            <span class="chip-file-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="8" y1="13" x2="16" y2="13"></line>
                    <line x1="8" y1="17" x2="16" y2="17"></line>
                </svg>
            </span>
            <span class="chip-name">{dataset_name}</span>
            <span class="chip-size">{size_str}</span>
        </div>
    </div>
    """


def export_dataset_to_temp(df: pd.DataFrame, base_name: str = "dataset") -> str:
    """Exports active DataFrame to a temporary downloadable CSV file."""
    if df is None:
        return None
    export_dir = os.path.join(tempfile.gettempdir(), "ai_data_analyst_exports")
    os.makedirs(export_dir, exist_ok=True)
    clean_base = os.path.splitext(os.path.basename(str(base_name or "dataset")))[0]
    export_path = os.path.join(export_dir, f"{clean_base}_processed.csv")
    df.to_csv(export_path, index=False)
    return export_path


# -------------------------------------------------------------
# Initial State Generator (Titanic Dataset as benchmark preview)
# -------------------------------------------------------------

def get_initial_demo_state():
    """Initializes the exact state shown in the reference design."""
    sample_path = os.path.join("datasets", "Titanic-Dataset.csv")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("datasets", "sample_data.csv")

    if os.path.exists(sample_path):
        df, _ = load_dataset(sample_path)
        dataset_name = os.path.basename(sample_path)
        file_size_kb = os.path.getsize(sample_path) / 1024.0
        overview = get_dataset_overview(df, dataset_name)
        profile = generate_full_profile(df)
        export_path = export_dataset_to_temp(df, dataset_name)

        num_cols = profile["column_types"]["numerical"]
        missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        default_impute = missing_cols[0] if missing_cols else (df.columns[0] if len(df.columns) > 0 else None)
        default_scale = num_cols[0] if num_cols else None

        banner_html = build_dataset_banner_html(dataset_name, overview, profile)
        chip_html = build_file_chip_html(dataset_name, file_size_kb)

        initial_chat = [
            {"role": "user", "content": "What is the age distribution of passengers in this dataset?"},
            {
                "role": "assistant",
                "content": (
                    "**Age Distribution Insights**\n"
                    "The age distribution is right-skewed, with most passengers between 15 and 40 years old.\n"
                    "• Mean age: 29.7 years\n"
                    "• Median age: 28.0 years\n"
                    "• Youngest: 0 years | Oldest: 80 years"
                ),
            },
        ]

        # Generate initial chart for Age if present
        target_col = "Age" if "Age" in df.columns else (num_cols[0] if num_cols else None)
        chart_fig = None
        if target_col:
            chart_fig, _ = create_histogram(df, target_col, bins=28, title="Age distribution")

        initial_context = [
            {
                "user": "What is the age distribution of passengers in this dataset?",
                "task": "visualize",
                "column": target_col,
                "analyzed_column": target_col,
            }
        ]

        return {
            "df": df,
            "name": dataset_name,
            "banner_html": banner_html,
            "chip_html": chip_html,
            "preview_df": df.head(10),
            "stats_df": profile["numerical_stats_table"],
            "missing_df": profile["missing_summary_table"],
            "chat": initial_chat,
            "context": initial_context,
            "plot": chart_fig,
            "export_path": export_path,
            "cols": list(df.columns),
            "num_cols": num_cols,
            "default_impute": default_impute,
            "default_scale": default_scale,
        }

    return {
        "df": None,
        "name": "",
        "banner_html": build_dataset_banner_html(None, None, None),
        "chip_html": "",
        "preview_df": None,
        "stats_df": None,
        "missing_df": None,
        "chat": [],
        "context": [],
        "plot": None,
        "export_path": None,
        "cols": [],
        "num_cols": [],
        "default_impute": None,
        "default_scale": None,
    }


# -------------------------------------------------------------
# Gradio Event Handlers
# -------------------------------------------------------------

def on_file_upload(file_obj):
    """Processes newly uploaded CSV or Excel dataset."""
    if file_obj is None:
        return (
            build_dataset_banner_html(None, None, None),  # banner
            "",  # chip
            None,  # preview_df
            None,  # numerical_stats_df
            None,  # missing_summary_df
            [],  # chatbot
            None,  # result_table
            None,  # result_plot
            None,  # active_df
            "",  # active_name
            gr.update(value=None, visible=False),  # download_btn
            gr.update(choices=[], value=None),  # impute_col
            gr.update(choices=[], value=None),  # scale_col
            [],  # context
        )

    file_path = file_obj.name
    dataset_name = os.path.basename(file_path)
    file_size_kb = os.path.getsize(file_path) / 1024.0

    df, err = load_dataset(file_path)
    if err:
        error_chat = [{"role": "assistant", "content": f"**Error Loading File**: {err}"}]
        return (
            build_dataset_banner_html(None, None, None),
            "",
            None,
            None,
            None,
            error_chat,
            None,
            None,
            None,
            "",
            gr.update(value=None, visible=False),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            [],
        )

    overview = get_dataset_overview(df, dataset_name=dataset_name)
    profile = generate_full_profile(df)
    export_path = export_dataset_to_temp(df, dataset_name)

    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    default_impute_col = missing_cols[0] if missing_cols else (list(df.columns)[0] if len(df.columns) > 0 else None)
    num_cols = profile["column_types"]["numerical"]
    default_scale_col = num_cols[0] if num_cols else None

    banner_html = build_dataset_banner_html(dataset_name, overview, profile)
    chip_html = build_file_chip_html(dataset_name, file_size_kb)

    welcome_chat = [
        {
            "role": "assistant",
            "content": f"Hello! I have loaded `{dataset_name}` ({overview['total_rows']:,} rows, {overview['total_columns']} columns). You can ask any question, request charts, or clean and scale features!",
        }
    ]

    return (
        banner_html,
        chip_html,
        df.head(10),
        profile["numerical_stats_table"],
        profile["missing_summary_table"],
        welcome_chat,
        None,
        None,
        df,
        dataset_name,
        gr.update(value=export_path, visible=True),
        gr.update(choices=list(df.columns), value=default_impute_col),
        gr.update(choices=num_cols, value=default_scale_col),
        [],
    )


def on_clear_dataset():
    """Clears loaded dataset and resets state."""
    plt.close("all")
    return (
        build_dataset_banner_html(None, None, None),  # banner
        "",  # chip
        None,  # preview_df
        None,  # numerical_stats_df
        None,  # missing_summary_df
        [],  # chatbot
        None,  # result_table
        None,  # result_plot
        "",  # query_input
        None,  # active_df
        "",  # active_name
        gr.update(value=None, visible=False),  # download_btn
        gr.update(choices=[], value=None),  # impute_col
        gr.update(choices=[], value=None),  # scale_col
        [],  # context
        None,  # file_input
    )


def on_ask_question(question, chat_history, context_history, active_df, active_name):
    """Processes user queries through query planner, Pandas engine, and Matplotlib."""
    if not question or not str(question).strip():
        return chat_history, context_history, None, None, "", active_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if active_df is None:
        new_hist = list(chat_history or [])
        new_hist.append({"role": "user", "content": question})
        new_hist.append({"role": "assistant", "content": "Please upload a CSV or Excel dataset first to analyze."})
        return new_hist, context_history, None, None, "", None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    current_df = active_df
    current_name = active_name or "dataset.csv"
    current_context = list(context_history or [])

    # Intent classification with scikit-learn
    ml_intent, ml_conf = classify_user_query(question)
    ml_tag = f"\n\n*[Intent: {ml_intent} ({ml_conf*100:.0f}% confidence)]*"

    # Plan query
    plan, plan_err = plan_query(question, current_df, conversation_history=current_context)
    if plan_err:
        new_hist = list(chat_history or [])
        new_hist.append({"role": "user", "content": question})
        new_hist.append({"role": "assistant", "content": f"**[Error]**: {plan_err}{ml_tag}"})
        return new_hist, current_context, None, None, "", current_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    # Execute deterministic plan in Pandas
    result = execute_plan(current_df, plan)

    # Track conversation context
    target_column = plan.get("column") or plan.get("group_by") or plan.get("condition_col")
    current_context.append({
        "user": question,
        "task": plan.get("task"),
        "column": target_column,
        "analyzed_column": target_column,
    })

    # Check if dataset was modified (impute / scale)
    modified = result.get("modified_df") is not None
    updated_df = result["modified_df"] if modified else current_df

    # Generate response
    response_text = generate_response(question, result, dataset_name=current_name, df=updated_df)
    full_response = f"{response_text}{ml_tag}"

    new_hist = list(chat_history or [])
    new_hist.append({"role": "user", "content": question})
    new_hist.append({"role": "assistant", "content": full_response})

    # Table output
    table_output = None
    if result.get("status") == "success" and isinstance(result.get("value"), pd.DataFrame):
        table_output = result["value"]

    # Chart output
    chart_fig = None
    if result.get("status") == "success":
        chart_fig, _ = create_auto_chart(result, updated_df)

    if modified:
        export_path = export_dataset_to_temp(updated_df, current_name)
        overview = get_dataset_overview(updated_df, dataset_name=current_name)
        profile = generate_full_profile(updated_df)
        num_cols = profile["column_types"]["numerical"]
        banner_html = build_dataset_banner_html(current_name, overview, profile)
        missing_cols = [c for c in updated_df.columns if updated_df[c].isnull().sum() > 0]
        col_update = gr.update(choices=list(updated_df.columns), value=missing_cols[0] if missing_cols else None)
        scale_update = gr.update(choices=num_cols, value=num_cols[0] if num_cols else None)

        return (
            new_hist,
            current_context,
            table_output,
            chart_fig,
            "",
            updated_df,
            banner_html,
            gr.skip(),  # chip stays
            updated_df.head(10),
            profile["numerical_stats_table"],
            profile["missing_summary_table"],
            gr.update(value=export_path, visible=True),
            col_update,
            scale_update,
        )

    return (
        new_hist,
        current_context,
        table_output,
        chart_fig,
        "",
        current_df,
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def on_apply_ui_impute(column, strategy_name, custom_val, active_df, active_name, chat_history, context_history):
    """Button click handler for missing value imputation."""
    if active_df is None or not column:
        return chat_history, context_history, None, None, active_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    strat_map = {
        "Mean (Numerical)": "mean",
        "Median (Numerical)": "median",
        "Mode (Most Frequent)": "mode",
        "Zero (0)": "zero",
        "Constant Value": "constant",
        "Drop Missing Rows": "drop",
    }
    strategy_code = strat_map.get(strategy_name, "mean")
    plan = {
        "task": "fill_missing",
        "column": column,
        "strategy": strategy_code,
        "fill_value": custom_val if custom_val else None,
    }

    result = execute_plan(active_df, plan)
    new_hist = list(chat_history or [])
    new_hist.append({"role": "user", "content": f"Fill missing values in '{column}' using {strategy_name}"})

    if result.get("status") == "success" and result.get("modified_df") is not None:
        df = result["modified_df"]
        export_path = export_dataset_to_temp(df, active_name)
        overview = get_dataset_overview(df, dataset_name=active_name)
        profile = generate_full_profile(df)
        num_cols = profile["column_types"]["numerical"]
        banner_html = build_dataset_banner_html(active_name, overview, profile)
        msg = result.get("message", "Imputation complete.")
        new_hist.append({"role": "assistant", "content": f"**Missing Values Updated**\n{msg}"})

        missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        col_update = gr.update(choices=list(df.columns), value=missing_cols[0] if missing_cols else None)
        scale_update = gr.update(choices=num_cols)

        return (
            new_hist,
            context_history,
            result.get("value"),
            None,
            df,
            banner_html,
            gr.skip(),
            df.head(10),
            profile["numerical_stats_table"],
            profile["missing_summary_table"],
            gr.update(value=export_path, visible=True),
            col_update,
            scale_update,
        )
    else:
        err = result.get("message") or result.get("error", "Failed to impute.")
        new_hist.append({"role": "assistant", "content": f"**Imputation Notice**: {err}"})
        return new_hist, context_history, None, None, active_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def on_apply_ui_scale(column, method_name, active_df, active_name, chat_history, context_history):
    """Button click handler for feature scaling and normalization."""
    if active_df is None or not column:
        return chat_history, context_history, None, None, active_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    method_code = "zscore"
    if "min-max" in method_name.lower():
        method_code = "minmax"
    elif "robust" in method_name.lower():
        method_code = "robust"

    plan = {
        "task": "scale_data",
        "column": column,
        "method": method_code,
    }

    result = execute_plan(active_df, plan)
    new_hist = list(chat_history or [])
    new_hist.append({"role": "user", "content": f"Scale '{column}' using {method_name}"})

    if result.get("status") == "success" and result.get("modified_df") is not None:
        df = result["modified_df"]
        export_path = export_dataset_to_temp(df, active_name)
        overview = get_dataset_overview(df, dataset_name=active_name)
        profile = generate_full_profile(df)
        num_cols = profile["column_types"]["numerical"]
        banner_html = build_dataset_banner_html(active_name, overview, profile)
        msg = result.get("message", "Feature scaled successfully.")
        new_hist.append({"role": "assistant", "content": f"**Feature Scaled**\n{msg}"})

        return (
            new_hist,
            context_history,
            result.get("value"),
            None,
            df,
            banner_html,
            gr.skip(),
            df.head(10),
            profile["numerical_stats_table"],
            profile["missing_summary_table"],
            gr.update(value=export_path, visible=True),
            gr.update(choices=list(df.columns)),
            gr.update(choices=num_cols, value=column if column in num_cols else (num_cols[0] if num_cols else None)),
        )
    else:
        err = result.get("message") or result.get("error", "Failed to scale.")
        new_hist.append({"role": "assistant", "content": f"**Scaling Notice**: {err}"})
        return new_hist, context_history, None, None, active_df, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


# -------------------------------------------------------------
# Custom CSS Theme (Dark Sleek Slate Design)
# -------------------------------------------------------------

custom_css = """
/* Reset & Global Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #070c18;
    --card-bg: #0d1629;
    --card-border: #1a2744;
    --card-header: #94a3b8;
    --accent-blue: #1d72f2;
    --accent-orange: #ff7a18;
    --accent-orange-hover: #ea580c;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --pill-bg: #111c34;
}

body, html, .gradio-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 1440px !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 24px 0 16px 0;
}
.header-badge-row {
    display: inline-flex;
    align-items: center;
    gap: 12px;
}
.header-robot-icon {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e3a8a, #0284c7);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
}
.header-title {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.header-title .bold-white {
    color: #f8fafc;
    font-weight: 800;
}
.header-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin: 4px 0 0 0;
    font-weight: 400;
}

/* Top Dataset Meta Bar */
.dataset-meta-bar {
    background: #0d1629;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.meta-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.file-sheet-icon {
    display: flex;
    align-items: center;
}
.file-name-text {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.meta-stats {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 13px;
    color: #94a3b8;
}
.meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.meta-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-green {
    background-color: #22c55e;
    box-shadow: 0 0 8px #22c55e;
}
.dot-gray {
    background-color: #64748b;
}
.meta-divider {
    width: 1px;
    height: 14px;
    background-color: #1e2c47;
}
.warning-pill {
    color: #fb923c;
}

/* Card Containers */
.panel-card {
    background: #0b1325 !important;
    border: 1px solid #182642 !important;
    border-radius: 12px !important;
    padding: 18px !important;
}
.panel-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* File Upload Component */
.upload-container {
    border: 1.5px dashed #203354 !important;
    border-radius: 10px !important;
    background: #091020 !important;
    padding: 18px !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
}
.upload-container:hover {
    border-color: #38bdf8 !important;
    background: #0a1326 !important;
}

/* Active File Chip */
.active-file-chip {
    background: #0d1629;
    border: 1px solid #1c2b48;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chip-info {
    display: flex;
    align-items: center;
    gap: 10px;
}
.chip-name {
    font-size: 13px;
    font-weight: 600;
    color: #f8fafc;
}
.chip-size {
    font-size: 12px;
    color: #64748b;
}

/* Preprocessing Mode Tabs */
.preproc-tabs .tab-nav {
    border-bottom: none !important;
    gap: 8px !important;
    margin-bottom: 14px !important;
}
.preproc-tabs .tab-nav button {
    flex: 1 !important;
    background: #111c34 !important;
    color: #94a3b8 !important;
    border: 1px solid #1c2b48 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
.preproc-tabs .tab-nav button.selected {
    background: #1d72f2 !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(29, 114, 242, 0.35) !important;
}

/* Form Controls */
label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
input, select, textarea {
    background: #091122 !important;
    border: 1px solid #1a2744 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 1px #38bdf8 !important;
}

/* Orange Action Buttons */
.btn-orange-action {
    background: linear-gradient(135deg, #ff7a18, #f97316) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.35) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.btn-orange-action:hover {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    box-shadow: 0 6px 18px rgba(249, 115, 22, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Download Button */
.btn-download-dark {
    background: #101c34 !important;
    border: 1px solid #1d2d4e !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.btn-download-dark:hover {
    background: #152444 !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
}

/* Quick Prompt Buttons Grid */
.quick-prompts-label {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    margin: 14px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.quick-prompt-btn {
    background: #101a2e !important;
    border: 1px solid #1c2b48 !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
}
.quick-prompt-btn:hover {
    background: #162440 !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Chatbot Styling */
.custom-chatbot {
    background: #091020 !important;
    border: 1px solid #16243e !important;
    border-radius: 10px !important;
}
.custom-chatbot .message.user {
    background: #121e38 !important;
    border: 1px solid #1e335a !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}
.custom-chatbot .message.bot {
    background: #0c1527 !important;
    border: 1px solid #182744 !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

/* Search / Query Input Bar */
.query-box-container input {
    background: #091020 !important;
    border: 1px solid #1a2744 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    padding: 12px 14px !important;
    font-size: 14px !important;
}

/* Output View Tabs */
.output-tabs .tab-nav {
    border-bottom: none !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
}
.output-tabs .tab-nav button {
    background: #101a2e !important;
    color: #94a3b8 !important;
    border: 1px solid #1c2b48 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.output-tabs .tab-nav button.selected {
    background: #1d72f2 !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}

/* Accordion Dark Theme */
.dark-accordion {
    background: #0c1427 !important;
    border: 1px solid #182642 !important;
    border-radius: 8px !important;
}
"""


# -------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------

initial_state = get_initial_demo_state()
api_status = check_api_status()

with gr.Blocks(title="AI Data Analyst Agent") as demo:
    # Inject CSS
    gr.HTML(f"<style>{custom_css}</style>")

    # Top Header
    gr.HTML(
        """
        <div class="header-container">
            <div class="header-badge-row">
                <div class="header-robot-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                        <circle cx="12" cy="5" r="2"></circle>
                        <path d="M12 7v4"></path>
                        <line x1="8" y1="16" x2="8" y2="16"></line>
                        <line x1="16" y1="16" x2="16" y2="16"></line>
                    </svg>
                </div>
                <h1 class="header-title">AI Data <span class="bold-white">Analyst Agent</span></h1>
            </div>
            <p class="header-subtitle">Conversational tabular analysis powered by Pandas, Scikit-learn & Ollama (Llama 3.2)</p>
        </div>
        """
    )

    # Dataset Status Banner (top bar)
    banner_component = gr.HTML(value=initial_state["banner_html"])

    # Application State Storage
    active_df_state = gr.State(initial_state["df"])
    active_name_state = gr.State(initial_state["name"])
    context_history = gr.State(initial_state["context"])

    with gr.Row():
        # =========================================================
        # LEFT COLUMN: DATASET & PREPROCESSING
        # =========================================================
        with gr.Column(scale=4, elem_classes=["panel-card"]):
            gr.HTML('<div class="panel-title">🗄️ DATASET & PREPROCESSING</div>')

            # Upload Component
            file_input = gr.File(
                label="Drop your CSV or Excel file here or click to upload",
                file_types=[".csv", ".xlsx", ".xls"],
                file_count="single",
                elem_classes=["upload-container"],
            )

            # Active file chip
            file_chip_component = gr.HTML(value=initial_state["chip_html"])

            # Preprocessing Tabs (Missing Values & Scaling)
            with gr.Tabs(elem_classes=["preproc-tabs"]):
                # Tab 1: Missing Values
                with gr.Tab("⚠️ Missing Values"):
                    with gr.Row():
                        impute_col_select = gr.Dropdown(
                            label="Target column ⓘ",
                            choices=initial_state["cols"],
                            value=initial_state["default_impute"],
                            interactive=True,
                            scale=5,
                        )
                        impute_strategy_select = gr.Dropdown(
                            label="Strategy ⓘ",
                            choices=["Mean (Numerical)", "Median (Numerical)", "Mode (Most Frequent)", "Zero (0)", "Constant Value", "Drop Missing Rows"],
                            value="Mean (Numerical)",
                            interactive=True,
                            scale=5,
                        )
                    impute_val_box = gr.Textbox(
                        label="Constant Value",
                        placeholder="e.g. Unknown or 0",
                        visible=False,
                    )
                    btn_apply_impute = gr.Button("⚡ Fill Missing Values", elem_classes=["btn-orange-action"])

                # Tab 2: Scaling
                with gr.Tab("📈 Scaling"):
                    with gr.Row():
                        scale_col_select = gr.Dropdown(
                            label="Target column ⓘ",
                            choices=initial_state["num_cols"],
                            value=initial_state["default_scale"],
                            interactive=True,
                            scale=5,
                        )
                        scale_method_select = gr.Dropdown(
                            label="Strategy ⓘ",
                            choices=["Standard (Z-Score)", "Min-Max (0 to 1)", "Robust (Median & IQR)"],
                            value="Standard (Z-Score)",
                            interactive=True,
                            scale=5,
                        )
                    btn_apply_scale = gr.Button("⚡ Apply Scaling", elem_classes=["btn-orange-action"])

            # Download Processed Dataset
            btn_download_file = gr.DownloadButton(
                "📥 Download Processed / Scaled Dataset (CSV)",
                value=initial_state["export_path"],
                elem_classes=["btn-download-dark"],
                visible=True if initial_state["export_path"] else False,
            )

            # Collapsible Dataset Profile & Statistics Accordion
            with gr.Accordion("📄 Dataset Profile & Statistics", open=False, elem_classes=["dark-accordion"]):
                with gr.Tab("Preview (10 Rows)"):
                    preview_df = gr.Dataframe(value=initial_state["preview_df"], interactive=False)
                with gr.Tab("Numerical Statistics"):
                    numerical_stats_df = gr.Dataframe(value=initial_state["stats_df"], interactive=False)
                with gr.Tab("Missing Breakdown"):
                    missing_summary_df = gr.Dataframe(value=initial_state["missing_df"], interactive=False)

            # Reset Button
            btn_reset_all = gr.Button("🔄 Clear / Reset Application", variant="stop", size="sm")

        # =========================================================
        # RIGHT COLUMN: ASK YOUR DATA
        # =========================================================
        with gr.Column(scale=6, elem_classes=["panel-card"]):
            gr.HTML('<div class="panel-title">💬 ASK YOUR DATA</div>')

            # Chatbot Container
            chatbot = gr.Chatbot(
                value=initial_state["chat"],
                elem_classes=["custom-chatbot"],
                height=300,
                show_label=False,
            )

            # Query Input Row
            with gr.Row(elem_classes=["query-box-container"]):
                query_input = gr.Textbox(
                    placeholder="Ask a question about your dataset...",
                    show_label=False,
                    scale=9,
                    container=False,
                )
                btn_send = gr.Button("✈ Ask", elem_classes=["btn-orange-action"], scale=2)

            # Quick Prompts Section
            gr.HTML('<div class="quick-prompts-label">⚡ Quick Prompts — General Tasks:</div>')
            with gr.Row():
                btn_missing = gr.Button("⚠️ Missing values", elem_classes=["quick-prompt-btn"])
                btn_corr = gr.Button("📊 Correlation matrix", elem_classes=["quick-prompt-btn"])
                btn_dist = gr.Button("📊 Distribution chart", elem_classes=["quick-prompt-btn"])
                btn_rows = gr.Button("📑 Total rows", elem_classes=["quick-prompt-btn"])
            with gr.Row():
                btn_cols = gr.Button("📋 Total columns", elem_classes=["quick-prompt-btn"])
                btn_dups = gr.Button("👥 Duplicate rows", elem_classes=["quick-prompt-btn"])
                btn_preview = gr.Button("👁️ Preview data", elem_classes=["quick-prompt-btn"])
                btn_top = gr.Button("📈 Top 5 records", elem_classes=["quick-prompt-btn"])

            # Output Views: Table & Visualization Tabs
            with gr.Tabs(elem_classes=["output-tabs"]):
                with gr.Tab("📈 Visualization"):
                    result_plot = gr.Plot(value=initial_state["plot"], show_label=False)
                with gr.Tab("📋 Table"):
                    result_table = gr.Dataframe(interactive=False, show_label=False)

    # -------------------------------------------------------------
    # Event Wiring
    # -------------------------------------------------------------

    # Impute Constant Value Box visibility
    impute_strategy_select.change(
        fn=lambda s: gr.update(visible=(s == "Constant Value")),
        inputs=[impute_strategy_select],
        outputs=[impute_val_box],
    )

    # File Upload Trigger
    file_input.change(
        fn=on_file_upload,
        inputs=[file_input],
        outputs=[
            banner_component,
            file_chip_component,
            preview_df,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            active_df_state,
            active_name_state,
            btn_download_file,
            impute_col_select,
            scale_col_select,
            context_history,
        ],
    )

    # Asking Questions
    ask_inputs = [query_input, chatbot, context_history, active_df_state, active_name_state]
    ask_outputs = [
        chatbot,
        context_history,
        result_table,
        result_plot,
        query_input,
        active_df_state,
        banner_component,
        file_chip_component,
        preview_df,
        numerical_stats_df,
        missing_summary_df,
        btn_download_file,
        impute_col_select,
        scale_col_select,
    ]

    btn_send.click(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)
    query_input.submit(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)

    # Imputation Trigger
    btn_apply_impute.click(
        fn=on_apply_ui_impute,
        inputs=[
            impute_col_select,
            impute_strategy_select,
            impute_val_box,
            active_df_state,
            active_name_state,
            chatbot,
            context_history,
        ],
        outputs=[
            chatbot,
            context_history,
            result_table,
            result_plot,
            active_df_state,
            banner_component,
            file_chip_component,
            preview_df,
            numerical_stats_df,
            missing_summary_df,
            btn_download_file,
            impute_col_select,
            scale_col_select,
        ],
    )

    # Scaling Trigger
    btn_apply_scale.click(
        fn=on_apply_ui_scale,
        inputs=[
            scale_col_select,
            scale_method_select,
            active_df_state,
            active_name_state,
            chatbot,
            context_history,
        ],
        outputs=[
            chatbot,
            context_history,
            result_table,
            result_plot,
            active_df_state,
            banner_component,
            file_chip_component,
            preview_df,
            numerical_stats_df,
            missing_summary_df,
            btn_download_file,
            impute_col_select,
            scale_col_select,
        ],
    )

    # Quick Prompts
    btn_missing.click(lambda: "Are there any missing values in this dataset?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_corr.click(lambda: "Show correlation between numerical columns", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_dist.click(lambda: "What is the age distribution of passengers in this dataset?", None, query_input).then(
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
    btn_preview.click(lambda: "Show preview of the first records", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_top.click(lambda: "Show the top 5 records", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )

    # Reset
    btn_reset_all.click(
        fn=on_clear_dataset,
        inputs=[],
        outputs=[
            banner_component,
            file_chip_component,
            preview_df,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            query_input,
            active_df_state,
            active_name_state,
            btn_download_file,
            impute_col_select,
            scale_col_select,
            context_history,
            file_input,
        ],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Data Analyst Agent Gradio Server...")
    print("=" * 60)
    demo.launch(server_name="127.0.0.1", share=False)
