"""
app.py
------
Main Gradio Web Application for AI Data Analyst Agent.
Integrates:
- File upload (CSV / Excel)
- Automated data profiling and schema summary
- Natural language query answering with Pandas
- Dynamic visualization rendering (Matplotlib)
- Conversation history tracking
"""

import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Import project modules
from data_loader import load_dataset, get_dataset_overview
from data_profiler import generate_full_profile
from query_planner import plan_query
from analysis_engine import execute_plan
from response_generator import generate_response
from visualization import create_auto_chart
from llm_client import check_api_status


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
        )

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

    welcome_chat = [
        {
            "role": "assistant",
            "content": f"Hello! I have loaded `{dataset_name}` with {overview['total_rows']} rows and {overview['total_columns']} columns. You can ask questions in English like 'What is the average salary?' or 'Show average salary by department'."
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
    )


def on_ask_question(question, file_obj, chat_history, context_history):
    """Processes user questions via the LLM/Heuristic planner and analysis engine."""
    if not question or not question.strip():
        return chat_history, context_history, None, None, ""

    if file_obj is None:
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": "Please upload a CSV or Excel dataset first."})
        return new_history, context_history, None, None, ""

    # Load dataset
    file_path = file_obj.name
    df, err = load_dataset(file_path)
    if err:
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": f"Dataset error: {err}"})
        return new_history, context_history, None, None, ""

    current_context = list(context_history or [])

    # 1. Classify Query Intent with Scikit-Learn ML Model (Viva ML Component)
    from query_classifier import classify_user_query
    ml_intent, ml_conf = classify_user_query(question)
    ml_badge = f"\n\n*[ML Intent Classified]*: `{ml_intent}` (Confidence: {ml_conf*100:.1f}%)"

    # 2. Translate question into validated plan with context
    plan, plan_err = plan_query(question, df, conversation_history=current_context)
    if plan_err:
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": f"**[Error]**: {plan_err}{ml_badge}"})
        return new_history, current_context, None, None, ""

    # 3. Execute plan with Pandas Analysis Engine
    result = execute_plan(df, plan)

    # 4. Record turn in contextual history for future follow-up questions
    target_column = plan.get("column") or plan.get("group_by") or plan.get("condition_col")
    current_context.append({
        "user": question,
        "task": plan.get("task"),
        "column": target_column,
        "analyzed_column": target_column,
    })

    # 5. Generate natural language response
    response_text = generate_response(question, result, dataset_name=os.path.basename(file_path))
    full_response = f"{response_text}{ml_badge}"

    new_history = list(chat_history or [])
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": full_response})

    # 6. Tabular result
    table_output = None
    if result.get("status") == "success" and isinstance(result.get("value"), pd.DataFrame):
        table_output = result["value"]

    # 7. Chart
    chart_fig = None
    if result.get("status") == "success":
        chart_fig, _ = create_auto_chart(result, df)

    return new_history, current_context, table_output, chart_fig, ""


def on_load_sample_dataset():
    """Loads the pre-packaged sample dataset for instant demonstration."""
    sample_path = os.path.join("datasets", "sample_data.csv")
    if not os.path.exists(sample_path):
        return None, "Sample dataset not found.", None, None, [], None, None

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
    )


# -------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------

api_status = check_api_status()
api_status_badge = (
    "🟢 **API Key**: Configured (Live Gemini LLM)"
    if api_status["configured"]
    else "🟡 **Mode**: Heuristic Fallback (Add Gemini API key in .env for full LLM power)"
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
                Natural-Language Conversational Tabular Analysis Powered by Pandas, Scikit-Learn & Gemini
            </p>
        </div>
        """
    )

    gr.Markdown(f"{api_status_badge} | **Supported Formats**: `.csv`, `.xlsx`, `.xls`")

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

            with gr.Row():
                query_input = gr.Textbox(
                    label="Ask a question about your data",
                    placeholder="e.g., What is the average salary? | Show salary by department | Top 3 employees",
                    scale=8,
                )
                btn_send = gr.Button("Ask", variant="primary", scale=2)

            # Quick Question Suggestions
            gr.Markdown("**Quick Prompts:**")
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

    # Conversational state tracking
    context_history = gr.State([])

    # File upload event (resets context history)
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
            context_history,
        ],
    )

    # Load sample dataset button (resets context history)
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
            context_history,
        ],
    )

    # Question asking triggers
    btn_send.click(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
    )
    query_input.submit(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
    )

    # Quick prompt buttons
    q1.click(lambda: "What is the average salary?", None, query_input).then(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
    )
    q2.click(lambda: "Show average salary by department", None, query_input).then(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
    )
    q3.click(lambda: "Top 3 employees by salary", None, query_input).then(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
    )
    q4.click(lambda: "Filter employees with salary greater than 70000", None, query_input).then(
        fn=on_ask_question,
        inputs=[query_input, file_input, chatbot, context_history],
        outputs=[chatbot, context_history, result_table, result_plot, query_input],
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
            context_history,
        ],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Data Analyst Agent Gradio Server...")
    print("=" * 60)
    # Launch Gradio server (automatically binds to 7860 or the next available port)
    demo.launch(server_name="127.0.0.1", share=False, css=custom_css)
