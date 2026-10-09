# 🤖 AI Data Analyst Agent

> **An LLM-Powered Conversational System for Automated Tabular Data Analysis**  
> *Developed for 3rd-Year B.E. (Artificial Intelligence & Machine Learning) Project Curriculum*

---

## 1. Project Overview

The **AI Data Analyst Agent** is a full-stack, conversational data analytics application that enables non-technical users and analysts to upload tabular datasets (**CSV** or **Excel**) and query them using natural English.

Unlike conventional LLM wrappers that ask models to calculate mathematical figures directly (which leads to hallucinations) or execute unverified LLM-generated code (which creates critical security risks), this application implements a **safe, decoupled architecture**:
1. An **NLP Machine Learning Classifier** (`Scikit-Learn`) categorizes the query intent.
2. An **LLM Query Planner** (`Google Gemini`) compiles the question into a structured JSON execution plan.
3. A **Deterministic Analysis Engine** (`Pandas` & `NumPy`) calculates the verified numerical answer.
4. An **Automated Visualization Engine** (`Matplotlib`) renders clean statistical plots.
5. An interactive **Web Application** (`Gradio`) displays conversational answers, summary tables, and charts.

---

## 2. Problem Statement

Business users and students often struggle to write complex SQL queries or Pandas commands to extract simple insights from tabular files. While general-purpose LLMs (like standard ChatGPT) can converse about data, they suffer from two major drawbacks:
1. **Arithmetic Hallucination**: LLMs predict text tokens; they do not perform arithmetic natively and frequently calculate incorrect averages, counts, or sums.
2. **Arbitrary Code Execution Hazards**: Systems running raw LLM-generated Python scripts via `eval()` or `exec()` expose hosts to prompt injection and accidental file loss.

**Solution**: This project bridges the gap by using LLMs only for **linguistic understanding and structured planning**, while delegating all computational execution to Pandas and NumPy.

---

## 3. Project Objectives

* Enable seamless drag-and-drop upload for `.csv` and `.xlsx` files with instant schema profiling.
* Perform automatic Exploratory Data Analysis (EDA): row/column counts, missing values, duplicate checks, and descriptive statistics.
* Implement a supervised **Machine Learning Intent Classifier** (TF-IDF + Logistic Regression) evaluating precision, recall, and F1-score across 11 intent classes.
* Translate free-form English queries into validated JSON execution plans.
* Safely compute 20 distinct tabular analytical tasks in Pandas without `eval()` or `exec()`.
* Automatically generate publication-grade Matplotlib charts (bar charts, histograms, scatter plots, trend lines).
* Maintain conversational context for follow-up questions (e.g. *"What is the maximum?"* after asking about salary).

---

## 4. Main Features

* **📁 Flexible File Ingestion**: Ingests CSV and Excel files, trims whitespace from column headers, and guards against empty or oversized files (>25 MB).
* **🔍 Automated Data Profiler**: Computes summary tables, separates numerical vs. categorical features, and identifies missing or duplicate data.
* **🧠 Classical ML Query Classifier**: 11-class NLP model providing instant category prediction with confidence scores.
* **🛡️ Sandboxed Analysis Engine**: Deterministic calculation engine covering mean, median, sum, min, max, standard deviation, filtering, sorting, ranking (Top/Bottom N), group-by aggregations, and correlation matrices.
* **📊 Dual-Mode Visualization**: Explicit visualization requests and smart auto-chart inference.
* **💬 Conversational Context**: Multi-turn dialogue history tracks previously analyzed columns for implicit follow-ups.
* **⚙️ Offline Fallback Mode**: Functions even without internet or an active API key using a rule-based heuristic planner.

---

## 5. Technology Stack

| Category | Technology / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core development language |
| **Data Processing** | `pandas` (>=2.2.0) | High-performance DataFrame operations |
| **Numerical Math** | `numpy` (>=1.26.0) | Numerical statistics and vector computations |
| **Excel Parser** | `openpyxl` (>=3.1.2) | Reading `.xlsx` workbook formats |
| **Machine Learning** | `scikit-learn` (>=1.4.0) | TF-IDF text vectorization, Logistic Regression classifier, and evaluation metrics |
| **Visualization** | `matplotlib` (>=3.8.0) | Clean figure rendering using server-safe `Agg` backend |
| **Web Interface** | `gradio` (>=4.20.0) | Reactive web UI with chat, tables, and plot rendering |
| **Environment Mgmt** | `python-dotenv` (>=1.0.1) | Secure API key isolation via `.env` |
| **LLM Integration** | `google-genai` (>=0.1.0) | Google Gemini 1.5/2.0 Flash official API SDK |

---

## 6. System Architecture

```text
               [ User Uploads CSV/Excel & Types Query in English ]
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │        Gradio Web User Interface        │
                   │                (app.py)                 │
                   └────────────────────┬────────────────────┘
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
┌──────────────────────────────┐                         ┌───────────────────────┐
│   Supervised ML Classifier   │                         │   LLM Query Planner   │
│    (query_classifier.py)     │                         │  (query_planner.py)   │
│   TF-IDF + Logistic Reg.     │                         │ Converts question to  │
│  Predicts 1 of 11 intents    │                         │  strict JSON plan     │
└──────────────┬───────────────┘                         └───────────┬───────────┘
               │                                                     │
               │ (Visual Badge)                                      ▼
               │                                         ┌───────────────────────┐
               │                                         │  Plan Safety Checker  │
               │                                         │ Schema & type checks  │
               │                                         └───────────┬───────────┘
               │                                                     │
               │                                                     ▼
               │                                         ┌───────────────────────┐
               │                                         │    Analysis Engine    │
               │                                         │ (analysis_engine.py)  │
               │                                         │  Pandas computations  │
               │                                         └─────┬───────────┬─────┘
               │                                               │           │
               │                        ┌──────────────────────┘           └──────────────────────┐
               │                        ▼                                                         ▼
               │          ┌───────────────────────────┐                             ┌───────────────────────────┐
               │          │    Response Generator     │                             │    Visualization Engine   │
               │          │  (response_generator.py)  │                             │    (visualization.py)     │
               │          │  Explains verified result │                             │ Generates Matplotlib plot │
               │          └─────────────┬─────────────┘                             └─────────────┬─────────────┘
               │                        │                                                         │
               └────────────────────────┼─────────────────────────────────────────────────────────┘
                                        ▼
                          [ Render Answer & Plot in UI ]
```

---

## 7. Project Folder Structure

```text
ai-data-analyst-agent/
│
├── app.py                      # Main Gradio application (UI & event orchestration)
├── data_loader.py              # Secure CSV & Excel loader with format & size guards
├── data_profiler.py            # Automated dataset profiling & descriptive statistics
├── llm_client.py               # Modular Google Gemini API integration with JSON mode
├── query_planner.py            # Converts natural language into validated JSON plans
├── analysis_engine.py          # Deterministic Pandas engine executing 20 analytical tasks
├── visualization.py            # Matplotlib figure generation (Bar, Line, Histogram, Scatter)
├── query_classifier.py         # Supervised ML model (TF-IDF + Logistic Regression)
├── response_generator.py       # Grounded conversational answer formulation
├── evaluation.py               # Empirical evaluation suite measuring correctness & latency
│
├── datasets/
│   ├── sample_data.csv         # Benchmark employee dataset with mixed features & nulls
│   └── sample_data.xlsx        # Excel workbook edition for format testing
│
├── tests/
│   ├── test_data_loader.py            # 5 tests: CSV/Excel loading, metadata, file errors
│   ├── test_data_profiler.py          # 6 tests: statistics, types, duplicate/null checks
│   ├── test_analysis_engine.py        # 10 tests: analytical tasks, bounds, edge cases
│   ├── test_llm_client.py             # 3 tests: status checks, JSON cleaning, error handling
│   ├── test_query_planner.py          # 5 tests: validation, heuristic planning, errors
│   ├── test_visualization.py          # 6 tests: bar, line, scatter, histogram rendering
│   ├── test_query_classifier.py       # 5 tests: model convergence, accuracy threshold
│   ├── test_conversational_followup.py# 3 tests: multi-turn dialogue, context inheritance
│   └── test_security_safeguards.py    # 4 tests: code injection, length limits, file types
│
├── evaluation_report.json      # Saved JSON report from evaluation benchmark
├── .env                        # Local API credentials (ignored by Git)
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules for venv, cache, and secrets
├── requirements.txt            # Minimal pip dependencies
└── README.md                   # Complete documentation and viva guide
```

---

## 8. Installation & Setup Instructions

### Step 1: Clone or Navigate to the Project
Open PowerShell or your VS Code terminal in the project directory:
```powershell
cd "c:\Users\yagni\OneDrive\Desktop\ML Project"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create venv
python -m venv venv

# Activate venv on Windows PowerShell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### Step 3: Install Dependencies
```powershell
.\venv\Scripts\pip install -r requirements.txt
```

### Step 4: Configure the Gemini API Key (Free)
1. Obtain a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Open [.env](file:///c:/Users/yagni/OneDrive/Desktop/ML%20Project/.env) in your editor and enter your key:
   ```ini
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   GEMINI_MODEL=gemini-1.5-flash
   ```
*(Note: If you run without an API key, the built-in Heuristic Planner will automatically take over, allowing full offline testing!)*

---

## 9. How to Run the Application

Start the local web server:
```powershell
.\venv\Scripts\python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:7860
```

### Quick UI Tour:
1. Click **"📊 Load Sample Employee Dataset"** (or drag and drop your own CSV/Excel file).
2. Expand **"📋 Dataset Profiling & Statistics"** to view descriptive statistics and missing values.
3. Use the one-click quick prompts or type any question into the chat box:
   * *"What is the average salary?"*
   * *"Show average salary by department"*
   * *"Top 3 employees by salary"*
   * *"Show employees with salary greater than 70000"*
   * *"Are there any missing values?"*
   * *"What is the correlation between experience and salary?"*

---

## 10. How to Train and Evaluate the ML Classifier

To execute the standalone Machine Learning intent classification pipeline:
```powershell
.\venv\Scripts\python query_classifier.py
```

### Measured Performance Metrics:
* **Algorithm**: Multiclass Logistic Regression (`solver='lbfgs'`, $C=5.0$)
* **Feature Extraction**: TF-IDF Vectorizer with Unigrams and Bigrams (`ngram_range=(1, 2)`)
* **Split Strategy**: Stratified Train/Test Split (75% train / 25% test, `random_state=42`)
* **Overall Test Accuracy**: **85.71%**
* **Macro F1-Score**: **0.8545**
* **Weighted F1-Score**: **0.8643**

---

## 11. How to Run the Automated Test Suite

The project includes **47 comprehensive automated unit tests** covering all modules:

```powershell
# Run the entire test suite:
.\venv\Scripts\python -m unittest discover tests

# Or run individual test modules:
.\venv\Scripts\python -m unittest tests/test_data_loader.py
.\venv\Scripts\python -m unittest tests/test_analysis_engine.py
.\venv\Scripts\python -m unittest tests/test_security_safeguards.py
```

---

## 12. Empirical Evaluation & Benchmark Results

Run the full evaluation benchmark:
```powershell
.\venv\Scripts\python evaluation.py
```

### Benchmark Summary (Measured from 12 Ground-Truth Test Cases):
* **Supported Analytical Queries**: 11
* **Analytical Correctness**: **100.0%** (11/11 verified numerical matches)
* **Execution Success Rate**: **100.0%**
* **Average Response Latency**: **30.1 milliseconds**
* **Out-of-Scope Safety Rejection**: **PASS** (Unsupported queries safely handled)

---

## 13. System Limitations

1. **Complex Multi-Table Joins**: The current release operates on single-table datasets. Multi-table relational joins (`JOIN` across multiple CSVs) are not yet automated.
2. **Extreme Class Imbalance on Novel Intents**: The ML query classifier was trained on 110 diverse seed questions across 11 classes; unusual phrasings may require expanding the training corpus.
3. **File Size Ceiling**: Uploads are restricted to 25 MB to maintain responsiveness in local memory.

---

## 14. Future Improvements

* **Multi-Table Relational Schema Support**: Allow uploading multiple CSVs and querying across foreign keys using DuckDB or SQLite.
* **AutoML Integration**: Incorporate automated model training (Random Forest, XGBoost) directly from the chat interface for predictive tasks.
* **Exportable PDF Reports**: Add a one-click button to export the generated charts, summary tables, and profile cards as a formatted PDF report.

---

## 15. Project Demonstration Plan (For College Evaluation)

When demonstrating this project to your professor, follow this 5-minute structured sequence:

1. **Show Architecture & Safe Execution (1 min)**:
   * Explain that unlike risky apps that execute `eval(llm_code)`, our system compiles queries into structured JSON and uses Pandas directly.
2. **Demonstrate File Loading & Profiling (1 min)**:
   * Click **"Load Sample Employee Dataset"**. Show the instant row/col count, missing value table, and descriptive statistics.
3. **Demonstrate NLP Data Analysis & Visualizations (1.5 min)**:
   * Ask: *"What is the average salary?"* $\rightarrow$ Point out the exact calculation ($67,333.33) and the ML Intent badge (`[aggregation]`).
   * Ask follow-up: *"What is the maximum?"* $\rightarrow$ Show conversational context in action (calculates $92,000 for Salary).
   * Ask: *"Show average salary by department"* $\rightarrow$ Show the resulting summary table and the generated Bar Chart with value labels.
4. **Demonstrate Classical Machine Learning (1 min)**:
   * Open the terminal and run `.\venv\Scripts\python query_classifier.py`.
   * Show the Stratified Split, Confusion Matrix, and 85.7% accuracy report.
5. **Demonstrate Test Suite & Benchmark (30 sec)**:
   * Run `.\venv\Scripts\python evaluation.py` to show 100% analytical correctness and 30 ms latency.

---

## 16. ML Viva Questions and Answers (Quick Revision)

### Q1: Why did you use TF-IDF instead of simple Count Vectorizer (Bag of Words)?
> **Answer**: Count Vectorizer only counts term occurrences, which gives disproportionate weight to frequent non-informative words (like *"what"*, *"is"*, *"the"*). **TF-IDF** multiplies Term Frequency by Inverse Document Frequency ($\log(N/DF)$). This penalizes ubiquitous words and assigns higher weights to domain-specific keywords (like *"average"*, *"histogram"*, *"correlation"*), dramatically improving classification accuracy.

### Q2: Why did you use Logistic Regression instead of Deep Learning or Random Forest for intent classification?
> **Answer**: For short text classification with small-to-medium labeled datasets, Logistic Regression with L2 regularization is computationally lightweight (<1 ms inference), does not require a GPU, is convex (guaranteed global optimum), and provides calibrated probability estimates through the Softmax function. Deep Learning models (like Transformers) would overfit on small corpora and introduce unnecessary overhead.

### Q3: Why is Stratified Split important when evaluating the classifier?
> **Answer**: In multiclass problems (we have 11 classes), a standard random split might accidentally assign all instances of a rare class to the training set and zero to the test set. A **Stratified Split** ensures that every class maintains the exact same percentage proportion in both the training and testing sets.

### Q4: Why can't we just ask the LLM to calculate the answers directly?
> **Answer**: Large Language Models are probabilistic autoregressive next-token predictors trained on text tokens. They do not have an internal arithmetic processor and regularly hallucinate numeric sums, averages, and group calculations. Delegating calculation to **Pandas and NumPy** guarantees 100% mathematical precision.

### Q5: How did you safeguard the system against arbitrary code execution attacks?
> **Answer**: We completely banned dynamic execution functions (`eval()` and `exec()`). The LLM Query Planner is restricted to producing strict JSON schemas with whitelisted tasks (`mean`, `filter`, `top_n`, etc.). The **Analysis Engine** acts as a deterministic dispatcher, mapping approved tasks to verified Pandas functions.
