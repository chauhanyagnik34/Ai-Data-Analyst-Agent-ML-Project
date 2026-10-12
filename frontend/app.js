/**
 * AI Data Analyst Agent (ADAA) - Modern Frontend Application Logic
 * Integrates live FastAPI backend APIs with instant interactive fallback engine.
 */

// Global State
const state = {
  currentDataset: null,
  activeDatasetId: null,
  activeDatasetName: "Titanic-Dataset.csv",
  activePage: "page-overview",
  isBackendConnected: false,
  llmStatus: { configured: false, model: "llama3.2", message: "Offline Heuristic Mode" },
  gridPage: 1,
  gridPageSize: 10,
  gridSearch: "",
  gridFilteredTotal: 0,
  gridSearchTimer: null,
  chatHistory: [],
  contextHistory: [],
  cleaningAudit: [],
  cachedCharts: {},
};

// =============================================================================
// 1. EMBEDDED SAMPLE DATA (Full Titanic & Employee Benchmark for 100% Offline Capability)
// =============================================================================

const TITANIC_SAMPLE_ROWS = [
  { PassengerId: 1, Survived: 0, Pclass: 3, Name: "Braund, Mr. Owen Harris", Sex: "male", Age: 22.0, SibSp: 1, Parch: 0, Ticket: "A/5 21171", Fare: 7.25, Cabin: null, Embarked: "S" },
  { PassengerId: 2, Survived: 1, Pclass: 1, Name: "Cumings, Mrs. John Bradley", Sex: "female", Age: 38.0, SibSp: 1, Parch: 0, Ticket: "PC 17599", Fare: 71.2833, Cabin: "C85", Embarked: "C" },
  { PassengerId: 3, Survived: 1, Pclass: 3, Name: "Heikkinen, Miss. Laina", Sex: "female", Age: 26.0, SibSp: 0, Parch: 0, Ticket: "STON/O2. 3101282", Fare: 7.925, Cabin: null, Embarked: "S" },
  { PassengerId: 4, Survived: 1, Pclass: 1, Name: "Futrelle, Mrs. Jacques Heath", Sex: "female", Age: 35.0, SibSp: 1, Parch: 0, Ticket: "113803", Fare: 53.1, Cabin: "C123", Embarked: "S" },
  { PassengerId: 5, Survived: 0, Pclass: 3, Name: "Allen, Mr. William Henry", Sex: "male", Age: 35.0, SibSp: 0, Parch: 0, Ticket: "373450", Fare: 8.05, Cabin: null, Embarked: "S" },
  { PassengerId: 6, Survived: 0, Pclass: 3, Name: "Moran, Mr. James", Sex: "male", Age: null, SibSp: 0, Parch: 0, Ticket: "330877", Fare: 8.4583, Cabin: null, Embarked: "Q" },
  { PassengerId: 7, Survived: 0, Pclass: 1, Name: "McCarthy, Mr. Timothy J", Sex: "male", Age: 54.0, SibSp: 0, Parch: 0, Ticket: "17463", Fare: 51.8625, Cabin: "E46", Embarked: "S" },
  { PassengerId: 8, Survived: 0, Pclass: 3, Name: "Palsson, Master. Gosta Leonard", Sex: "male", Age: 2.0, SibSp: 3, Parch: 1, Ticket: "349909", Fare: 21.075, Cabin: null, Embarked: "S" },
  { PassengerId: 9, Survived: 1, Pclass: 3, Name: "Johnson, Mrs. Oscar W", Sex: "female", Age: 27.0, SibSp: 0, Parch: 2, Ticket: "347742", Fare: 11.1333, Cabin: null, Embarked: "S" },
  { PassengerId: 10, Survived: 1, Pclass: 2, Name: "Nasser, Mrs. Nicholas", Sex: "female", Age: 14.0, SibSp: 1, Parch: 0, Ticket: "237736", Fare: 30.0708, Cabin: null, Embarked: "C" },
  { PassengerId: 11, Survived: 1, Pclass: 3, Name: "Sandstrom, Miss. Marguerite Rut", Sex: "female", Age: 4.0, SibSp: 1, Parch: 1, Ticket: "PP 9549", Fare: 16.7, Cabin: "G6", Embarked: "S" },
  { PassengerId: 12, Survived: 1, Pclass: 1, Name: "Bonnell, Miss. Elizabeth", Sex: "female", Age: 58.0, SibSp: 0, Parch: 0, Ticket: "113783", Fare: 26.55, Cabin: "C103", Embarked: "S" },
  { PassengerId: 13, Survived: 0, Pclass: 3, Name: "Saundercock, Mr. William Henry", Sex: "male", Age: 20.0, SibSp: 0, Parch: 0, Ticket: "A/5. 2151", Fare: 8.05, Cabin: null, Embarked: "S" },
  { PassengerId: 14, Survived: 0, Pclass: 3, Name: "Andersson, Mr. Anders Johan", Sex: "male", Age: 39.0, SibSp: 1, Parch: 5, Ticket: "347082", Fare: 31.275, Cabin: null, Embarked: "S" },
  { PassengerId: 15, Survived: 0, Pclass: 3, Name: "Vestrom, Miss. Hulda Amanda Adolfina", Sex: "female", Age: 14.0, SibSp: 0, Parch: 0, Ticket: "350406", Fare: 7.8542, Cabin: null, Embarked: "S" },
];

const EMPLOYEE_SAMPLE_ROWS = [
  { EmployeeID: 101, Name: "Aarav Sharma", Department: "Engineering", Role: "Backend Developer", Salary: 75000, Experience_Years: 4, Performance_Score: 4.2, Projects_Completed: 8, Join_Date: "2021-03-15" },
  { EmployeeID: 102, Name: "Diya Patel", Department: "Engineering", Role: "Frontend Developer", Salary: 68000, Experience_Years: 3, Performance_Score: 4.5, Projects_Completed: 7, Join_Date: "2022-01-10" },
  { EmployeeID: 103, Name: "Rohan Verma", Department: "Sales", Role: "Sales Executive", Salary: 52000, Experience_Years: 2, Performance_Score: 3.8, Projects_Completed: 12, Join_Date: "2023-05-20" },
  { EmployeeID: 104, Name: "Sneha Iyer", Department: "Marketing", Role: "Content Strategist", Salary: 58000, Experience_Years: 3, Performance_Score: 4.0, Projects_Completed: 9, Join_Date: "2022-08-11" },
  { EmployeeID: 105, Name: "Vikram Singh", Department: "Finance", Role: "Financial Analyst", Salary: 82000, Experience_Years: 6, Performance_Score: 4.6, Projects_Completed: 11, Join_Date: "2019-11-04" },
  { EmployeeID: 106, Name: "Ananya Roy", Department: "HR", Role: "HR Specialist", Salary: 48000, Experience_Years: 2, Performance_Score: 3.9, Projects_Completed: 5, Join_Date: "2023-02-18" },
  { EmployeeID: 107, Name: "Karan Gupta", Department: "Engineering", Role: "DevOps Engineer", Salary: 92000, Experience_Years: 7, Performance_Score: 4.8, Projects_Completed: 15, Join_Date: "2018-07-22" },
  { EmployeeID: 108, Name: "Pooja Nair", Department: "Sales", Role: "Sales Manager", Salary: 88000, Experience_Years: 8, Performance_Score: 4.4, Projects_Completed: 18, Join_Date: "2017-09-30" },
  { EmployeeID: 109, Name: "Rahul Mehta", Department: "Engineering", Role: "Full Stack Developer", Salary: 79000, Experience_Years: 5, Performance_Score: null, Projects_Completed: 10, Join_Date: "2020-12-01" },
  { EmployeeID: 110, Name: "Ishita Das", Department: "Finance", Role: "Senior Accountant", Salary: 71000, Experience_Years: 5, Performance_Score: 4.1, Projects_Completed: 8, Join_Date: "2021-06-14" },
  { EmployeeID: 111, Name: "Aditya Joshi", Department: "Marketing", Role: "SEO Specialist", Salary: 50000, Experience_Years: 2, Performance_Score: 3.7, Projects_Completed: 6, Join_Date: "2023-04-10" },
  { EmployeeID: 112, Name: "Neha Kulkarni", Department: "HR", Role: "Recruiting Coordinator", Salary: 46000, Experience_Years: 1, Performance_Score: 3.6, Projects_Completed: 4, Join_Date: "2024-01-15" },
  { EmployeeID: 113, Name: "Siddharth Sen", Department: "Engineering", Role: "Data Engineer", Salary: 85000, Experience_Years: 5, Performance_Score: 4.7, Projects_Completed: 13, Join_Date: "2020-04-25" },
  { EmployeeID: 114, Name: "Kavita Rao", Department: "Sales", Role: "Account Executive", Salary: 54000, Experience_Years: 3, Performance_Score: 3.9, Projects_Completed: 10, Join_Date: "2022-10-05" },
  { EmployeeID: 115, Name: "Manish Pandey", Department: "Engineering", Role: "QA Engineer", Salary: 62000, Experience_Years: 3, Performance_Score: 4.0, Projects_Completed: 9, Join_Date: "2022-03-19" },
];

function createDefaultDataset(key = "titanic") {
  if (key === "employee") {
    return {
      id: "preset-employee",
      name: "sample_data.csv",
      row_count: 15,
      column_count: 9,
      headers: ["EmployeeID", "Name", "Department", "Role", "Salary", "Experience_Years", "Performance_Score", "Projects_Completed", "Join_Date"],
      columns: [
        { name: "EmployeeID", type: "Numeric", dtype: "int64", missing: 0, unique: 15 },
        { name: "Name", type: "Categorical", dtype: "object", missing: 0, unique: 15 },
        { name: "Department", type: "Categorical", dtype: "object", missing: 0, unique: 4 },
        { name: "Role", type: "Categorical", dtype: "object", missing: 0, unique: 14 },
        { name: "Salary", type: "Numeric", dtype: "int64", missing: 0, unique: 15 },
        { name: "Experience_Years", type: "Numeric", dtype: "int64", missing: 0, unique: 8 },
        { name: "Performance_Score", type: "Numeric", dtype: "float64", missing: 1, unique: 10 },
        { name: "Projects_Completed", type: "Numeric", dtype: "int64", missing: 0, unique: 12 },
        { name: "Join_Date", type: "Date", dtype: "object", missing: 0, unique: 15 },
      ],
      preview_rows: EMPLOYEE_SAMPLE_ROWS,
      missing_count: 1,
      missing_percentage: 0.74,
      completeness: 99.26,
      duplicate_count: 0,
      quality_score: 98.2,
      memory_usage: "1.1 KB",
      column_types: {
        numerical: ["EmployeeID", "Salary", "Experience_Years", "Performance_Score", "Projects_Completed"],
        categorical: ["Name", "Department", "Role"],
        datetime: ["Join_Date"]
      },
      identifier_columns: ["EmployeeID"],
      numeric_stats: [
        { Column: "Salary", Count: 15, Mean: 67333.33, Std: 15159.47, Min: 46000, "25%": 53000, "50%": 68000, "75%": 80500, Max: 92000 },
        { Column: "Experience_Years", Count: 15, Mean: 4.13, Std: 2.07, Min: 1, "25%": 2.5, "50%": 4, "75%": 5.5, Max: 8 },
        { Column: "Performance_Score", Count: 14, Mean: 4.19, Std: 0.38, Min: 3.6, "25%": 3.9, "50%": 4.15, "75%": 4.5, Max: 4.8 },
        { Column: "Projects_Completed", Count: 15, Mean: 10.27, Std: 3.95, Min: 4, "25%": 7.5, "50%": 10, "75%": 12.5, Max: 18 }
      ],
      missing_summary: [
        { Column: "Performance_Score", Missing_Count: 1, Missing_Percentage: 6.67, Data_Type: "float64" },
        { Column: "Salary", Missing_Count: 0, Missing_Percentage: 0.0, Data_Type: "int64" }
      ]
    };
  }

  // Titanic default
  return {
    id: "preset-titanic",
    name: "Titanic-Dataset.csv",
    row_count: 891,
    column_count: 12,
    headers: ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
    columns: [
      { name: "PassengerId", type: "Numeric", dtype: "int64", missing: 0, unique: 891 },
      { name: "Survived", type: "Numeric", dtype: "int64", missing: 0, unique: 2 },
      { name: "Pclass", type: "Numeric", dtype: "int64", missing: 0, unique: 3 },
      { name: "Name", type: "Categorical", dtype: "object", missing: 0, unique: 891 },
      { name: "Sex", type: "Categorical", dtype: "object", missing: 0, unique: 2 },
      { name: "Age", type: "Numeric", dtype: "float64", missing: 177, unique: 88 },
      { name: "SibSp", type: "Numeric", dtype: "int64", missing: 0, unique: 7 },
      { name: "Parch", type: "Numeric", dtype: "int64", missing: 0, unique: 7 },
      { name: "Ticket", type: "Categorical", dtype: "object", missing: 0, unique: 681 },
      { name: "Fare", type: "Numeric", dtype: "float64", missing: 0, unique: 248 },
      { name: "Cabin", type: "Categorical", dtype: "object", missing: 687, unique: 147 },
      { name: "Embarked", type: "Categorical", dtype: "object", missing: 2, unique: 3 },
    ],
    preview_rows: TITANIC_SAMPLE_ROWS,
    missing_count: 866,
    missing_percentage: 8.09,
    completeness: 91.91,
    duplicate_count: 0,
    quality_score: 87.4,
    memory_usage: "83.7 KB",
    column_types: {
      numerical: ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"],
      categorical: ["Name", "Sex", "Ticket", "Cabin", "Embarked"],
      datetime: []
    },
    identifier_columns: ["PassengerId"],
    numeric_stats: [
      { Column: "Age", Count: 714, Mean: 29.70, Std: 14.53, Min: 0.42, "25%": 20.12, "50%": 28.0, "75%": 38.0, Max: 80.0 },
      { Column: "Fare", Count: 891, Mean: 32.20, Std: 49.69, Min: 0.0, "25%": 7.91, "50%": 14.45, "75%": 31.0, Max: 512.33 },
      { Column: "Survived", Count: 891, Mean: 0.38, Std: 0.49, Min: 0, "25%": 0, "50%": 0, "75%": 1, Max: 1 },
      { Column: "Pclass", Count: 891, Mean: 2.31, Std: 0.84, Min: 1, "25%": 2, "50%": 3, "75%": 3, Max: 3 }
    ],
    missing_summary: [
      { Column: "Cabin", Missing_Count: 687, Missing_Percentage: 77.10, Data_Type: "object" },
      { Column: "Age", Missing_Count: 177, Missing_Percentage: 19.87, Data_Type: "float64" },
      { Column: "Embarked", Missing_Count: 2, Missing_Percentage: 0.22, Data_Type: "object" }
    ]
  };
}

// =============================================================================
// 2. INITIALIZATION & BACKEND SYNC
// =============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  applyTheme(localStorage.getItem("adaa-theme") || "dark");
  setupNavigation();
  setupEventListeners();

  // Connect to the local FastAPI backend before showing dataset values.
  await checkBackendConnection();
  if (state.isBackendConnected) {
    const restored = await restoreSavedDataset();
    if (!restored) await loadPresetDataset("titanic");
    loadIntentMetrics();
  } else {
    setDatasetState(null);
    setUploadStatus("app.py is not connected. Start the project with python app.py to upload and analyze data.");
  }

  const requestedPage = window.location.hash.slice(1);
  if (requestedPage && document.getElementById(requestedPage)) switchPage(requestedPage, false);
});

async function checkBackendConnection() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      state.isBackendConnected = true;
      state.llmStatus = {
        configured: data.llm_connected,
        model: data.llm_model || "llama3.2",
        message: data.llm_message || "Online",
      };
      updateConnectionBadges(true);
      return true;
    }
  } catch (err) {
    // Backend offline / static HTML preview mode
  }
  state.isBackendConnected = false;
  updateConnectionBadges(false);
  return false;
}

function updateConnectionBadges(isConnected) {
  const connDot = document.getElementById("connectionDot");
  const connText = document.getElementById("connectionText");
  const llmBadge = document.getElementById("llmStatusBadge");

  if (isConnected) {
    if (connDot) connDot.style.background = "var(--accent-emerald)";
    if (connText) connText.textContent = "FastAPI Backend Live";
    if (llmBadge) {
      if (state.llmStatus.configured) {
        llmBadge.innerHTML = `🟢 <b>LLM</b>: Connected (Ollama - ${state.llmStatus.model})`;
        llmBadge.className = "badge-pill badge-id";
      } else {
        llmBadge.innerHTML = `🟡 <b>Mode</b>: Safe Heuristic Fallback`;
        llmBadge.className = "badge-pill badge-datetime";
      }
    }
  } else {
    if (connDot) connDot.style.background = "var(--accent-amber)";
    if (connText) connText.textContent = "Interactive Demo Engine";
    if (llmBadge) {
      llmBadge.innerHTML = `⚡ <b>Client Demo</b>: Built-in Benchmark Data`;
      llmBadge.className = "badge-pill badge-datetime";
    }
  }
}

// =============================================================================
// 3. NAVIGATION & VIEW SWITCHING
// =============================================================================

function setupNavigation() {
  const navLinks = document.querySelectorAll(".nav-link[data-page]");
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetPage = link.getAttribute("data-page");
      switchPage(targetPage);
      document.querySelector(".app-sidebar")?.classList.remove("open");
    });
  });
}

function switchPage(pageId, updateLocation = true) {
  if (!document.getElementById(pageId)) return;
  state.activePage = pageId;
  if (updateLocation && window.location.hash !== `#${pageId}`) {
    window.history.replaceState(null, "", `#${pageId}`);
  }

  // Update nav links
  document.querySelectorAll(".nav-link").forEach((link) => {
    if (link.getAttribute("data-page") === pageId) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  // Update view sections
  document.querySelectorAll(".view-section").forEach((sec) => {
    if (sec.id === pageId) {
      sec.classList.add("active");
    } else {
      sec.classList.remove("active");
    }
  });

  // Trigger page-specific renders
  onPageActivated(pageId);
}

function applyTheme(theme) {
  const nextTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = nextTheme;
  const isLight = nextTheme === "light";
  const icon = document.getElementById("themeIcon");
  const label = document.getElementById("themeLabel");
  const button = document.getElementById("btnThemeToggle");
  if (icon) icon.textContent = isLight ? "🌙" : "☀️";
  if (label) label.textContent = isLight ? "Dark mode" : "Light mode";
  if (button) button.setAttribute("aria-pressed", String(isLight));
  localStorage.setItem("adaa-theme", nextTheme);
}

function onPageActivated(pageId) {
  if (!state.currentDataset) return;

  switch (pageId) {
    case "page-overview":
      renderOverviewTab();
      break;
    case "page-selection":
      populateSelectionControls();
      break;
    case "page-insights":
      populateAdvancedControls();
      break;
    case "page-preprocessing":
      populatePreprocessingControls();
      break;
    case "page-transformation":
      populateTransformationControls();
      break;
    case "page-mining":
      populateMiningControls();
      break;
    case "page-visualization":
      populateVisualizationControls();
      break;
    case "page-ai":
      populateAIPrompts();
      break;
    default:
      break;
  }
}

// =============================================================================
// 4. DATASET MANAGEMENT & PRESETS
// =============================================================================

async function loadPresetDataset(key) {
  if (!state.isBackendConnected) {
    showToast("Start app.py to load project datasets.", "error");
    return false;
  }
  showToast(`Loading dataset '${key}'...`, "info");
  try {
    const res = await fetch(`/api/datasets/presets/${encodeURIComponent(key)}`, { method: "POST" });
    if (!res.ok) throw new Error(await responseError(res));
    const payload = await res.json();
    setDatasetState(payload);
    setUploadStatus(`${payload.name} loaded · ${payload.row_count.toLocaleString()} rows · ${payload.column_count} columns.`);
    showToast(`Loaded ${payload.name} (${payload.row_count.toLocaleString()} rows)`, "success");
    return true;
  } catch (err) {
    showToast(err.message || "Could not load the sample dataset.", "error");
    setUploadStatus(err.message || "Could not load the sample dataset.");
    return false;
  }
}

async function restoreSavedDataset() {
  const savedId = localStorage.getItem("adaa-dataset-id");
  if (!savedId) return false;
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(savedId)}`);
    if (!res.ok) throw new Error("Saved dataset session expired.");
    setDatasetState(await res.json());
    setUploadStatus(`${state.currentDataset.name} restored from this browser session.`);
    return true;
  } catch (_err) {
    localStorage.removeItem("adaa-dataset-id");
    return false;
  }
}

function setUploadStatus(message, isError = false) {
  const status = document.getElementById("uploadStatus");
  if (!status) return;
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function uploadDataset(file) {
  if (!file) return;
  const allowedExtensions = [".csv", ".xlsx", ".xls"];
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowedExtensions.includes(extension)) {
    setUploadStatus("Choose a CSV, XLSX, or XLS file.", true);
    showToast("Unsupported file type. Use CSV or Excel.", "error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    setUploadStatus("This file is larger than the 25 MB upload limit.", true);
    showToast("Files must be 25 MB or smaller.", "error");
    return;
  }
  if (!state.isBackendConnected) {
    setUploadStatus("app.py is not connected. Start the project with python app.py first.", true);
    showToast("Connect to app.py before uploading.", "error");
    return;
  }

  const zone = document.getElementById("datasetDropzone");
  zone?.classList.add("uploading");
  setUploadStatus(`Uploading ${file.name}…`);
  try {
    const res = await fetch(`/api/datasets?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    if (!res.ok) throw new Error(await responseError(res));
    const payload = await res.json();
    setDatasetState(payload);
    setUploadStatus(`${file.name} loaded · ${payload.row_count.toLocaleString()} rows · ${payload.column_count} columns.`);
    showToast(`Uploaded ${file.name}`, "success");
  } catch (err) {
    setUploadStatus(err.message || "Upload failed. Check the file and try again.", true);
    showToast(err.message || "Upload failed.", "error");
  } finally {
    zone?.classList.remove("uploading");
    const input = document.getElementById("datasetFileInput");
    if (input) input.value = "";
  }
}

async function responseError(response) {
  try {
    const data = await response.json();
    return data.detail || data.message || `Request failed (${response.status}).`;
  } catch (_err) {
    return `Request failed (${response.status}).`;
  }
}

function setDatasetState(data) {
  state.currentDataset = data || null;
  state.activeDatasetId = data?.id || null;
  state.activeDatasetName = data?.name || "No dataset loaded";
  state.gridPage = 1;
  state.gridFilteredTotal = data?.row_count || 0;
  state.gridSearch = "";
  const searchInput = document.getElementById("gridSearchInput");
  if (searchInput) searchInput.value = "";
  if (data?.id && !String(data.id).startsWith("preset-")) {
    localStorage.setItem("adaa-dataset-id", data.id);
  } else if (!data) {
    localStorage.removeItem("adaa-dataset-id");
  }

  // Update Header Badges
  const badgeName = document.getElementById("headerDatasetName");
  const badgeMeta = document.getElementById("headerDatasetMeta");
  if (badgeName) badgeName.textContent = data?.name || "No dataset loaded";
  if (badgeMeta) badgeMeta.textContent = data
    ? `${data.row_count.toLocaleString()} rows • ${data.column_count} cols • ${data.quality_score}% quality`
    : "Upload a file or load a sample dataset";

  // Render active page
  if (data) {
    onPageActivated(state.activePage);
    updateDatasetSummary(data);
  } else {
    updateDatasetSummary(null);
    clearDatasetViews();
  }
}

function updateDatasetSummary(data) {
  const story = document.getElementById("datasetStory");
  const types = document.getElementById("datasetTypeSummary");
  const notes = document.getElementById("profileNotes");
  if (story) story.textContent = data?.about_data?.summary || "Load a dataset to see its row count, features, and overall quality summary.";
  if (types) {
    const counts = data?.about_data || {};
    types.innerHTML = data ? `
      <span class="type-stat"><b>${Number(counts.numerical_count || 0)}</b> numeric</span>
      <span class="type-stat"><b>${Number(counts.categorical_count || 0)}</b> categorical</span>
      <span class="type-stat"><b>${Number(counts.datetime_count || 0)}</b> date/time</span>
    ` : "";
  }
  if (notes) {
    notes.textContent = data
      ? `${data.column_count} columns · ${data.memory_usage || "Memory size unavailable"} in memory · ${data.completeness}% complete · ${data.duplicate_count} duplicate rows.`
      : "Numeric statistics, column types, and missing-value summaries will appear here after loading data.";
  }
}

function clearDatasetViews() {
  ["schemaTable", "missingSummaryTable", "numericalStatsTable", "selectionResultTable", "cleaningResultTable", "transResultTable", "miningResultTable", "studioChartTable"]
    .forEach((id) => {
      const table = document.getElementById(id);
      if (table) { table.querySelector("thead")?.replaceChildren(); table.querySelector("tbody")?.replaceChildren(); }
    });
  const preview = document.querySelector("#gridDataTable tbody");
  if (preview) preview.innerHTML = '<tr><td class="empty-state-cell">Upload a dataset or load a sample to preview rows.</td></tr>';
  const chart = document.getElementById("studioChartResult");
  if (chart) chart.textContent = "Load a dataset to generate charts.";
}

// =============================================================================
// 5. PAGE 1: OVERVIEW & PROFILER TAB
// =============================================================================

function renderOverviewTab() {
  const d = state.currentDataset;
  if (!d) return;

  // KPI Metrics
  document.getElementById("kpiTotalRows").textContent = Number(d.row_count || 0).toLocaleString();
  document.getElementById("kpiTotalCols").textContent = Number(d.column_count || 0).toLocaleString();
  document.getElementById("kpiMissingCells").textContent = `${Number(d.missing_count || 0).toLocaleString()} (${Number(d.missing_percentage || 0)}%)`;
  document.getElementById("kpiCompleteness").textContent = `${Number(d.completeness || 0)}%`;
  document.getElementById("kpiDuplicates").textContent = Number(d.duplicate_count || 0).toLocaleString();
  document.getElementById("kpiQualityScore").textContent = `${Number(d.quality_score || 0)} / 100`;

  // Populate Schema Table
  const schemaTbody = document.querySelector("#schemaTable tbody");
  if (schemaTbody) {
    schemaTbody.innerHTML = "";
    (d.columns || []).forEach((c) => {
      const tr = document.createElement("tr");
      let typeBadge = `<span class="badge-pill badge-numeric">Numeric</span>`;
      if (c.type === "Categorical") typeBadge = `<span class="badge-pill badge-categorical">Categorical</span>`;
      if (c.type === "Date") typeBadge = `<span class="badge-pill badge-datetime">Datetime</span>`;
      if ((d.identifier_columns || []).includes(c.name)) typeBadge = `<span class="badge-pill badge-id">Primary ID</span>`;
      if (c.name === "Survived") typeBadge = `<span class="badge-pill badge-target">Target Label</span>`;

      tr.innerHTML = `
        <td style="font-weight:600;">${escapeHTML(c.name)}</td>
        <td>${typeBadge}</td>
        <td><code>${escapeHTML(c.dtype)}</code></td>
        <td style="color:${c.missing > 0 ? 'var(--accent-amber)' : 'var(--text-muted)'};">${Number(c.missing || 0).toLocaleString()}</td>
        <td>${Number(c.unique || 0).toLocaleString()}</td>
      `;
      schemaTbody.appendChild(tr);
    });
  }

  // Populate Numerical Stats
  const numTbody = document.querySelector("#numericalStatsTable tbody");
  if (numTbody) {
    numTbody.innerHTML = "";
    (d.numeric_stats || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:600;">${escapeHTML(row.Column || row.index || "")}</td>
        <td>${row.Count ?? '-'}</td>
        <td>${formatNumber(row.Mean)}</td>
        <td>${formatNumber(row["Std Dev"] ?? row.Std)}</td>
        <td>${formatNumber(row.Min)}</td>
        <td>${formatNumber(row["50%"] ?? row.Median)}</td>
        <td>${formatNumber(row.Max)}</td>
        <td>${formatNumber(row.Skewness)}</td>
        <td>${formatNumber(row["Outliers (IQR)"])}</td>
      `;
      numTbody.appendChild(tr);
    });
  }

  // Populate Missing Summary
  const missTbody = document.querySelector("#missingSummaryTable tbody");
  if (missTbody) {
    missTbody.innerHTML = "";
    (d.missing_summary || []).forEach((row) => {
      const tr = document.createElement("tr");
      const missingCount = row.Missing_Count ?? row["Missing Count"] ?? 0;
      const missingPct = row.Missing_Percentage ?? row["Missing Percentage (%)"] ?? 0;
      const dataType = row.Data_Type ?? row["Data Type"] ?? (d.columns || []).find((column) => column.name === row.Column)?.dtype ?? "";
      const pct = Math.min(100, Math.max(0, parseFloat(missingPct || 0)));
      let barColor = pct > 50 ? 'var(--accent-rose)' : (pct > 15 ? 'var(--accent-amber)' : 'var(--accent-cyan)');
      tr.innerHTML = `
        <td style="font-weight:600;">${escapeHTML(row.Column || "")}</td>
        <td>${Number(missingCount).toLocaleString()}</td>
        <td>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="flex:1; height:6px; background:#1e293b; border-radius:3px; overflow:hidden;">
              <div style="width:${pct}%; height:100%; background:${barColor};"></div>
            </div>
            <span>${pct}%</span>
          </div>
        </td>
        <td><code>${escapeHTML(dataType)}</code></td>
      `;
      missTbody.appendChild(tr);
    });
  }

  // Render Data Preview Grid
  if (state.isBackendConnected && state.activeDatasetId) loadPreviewPage(state.gridPage || 1);
  else renderDataGrid(d.preview_rows, d.headers, d.preview_rows?.length || 0, 0);
}

function renderDataGrid(rows, headers, total = (rows || []).length, offset = 0) {
  const thead = document.querySelector("#gridDataTable thead");
  const tbody = document.querySelector("#gridDataTable tbody");
  if (!thead || !tbody) return;

  // Header row
  thead.innerHTML = "";
  const trHead = document.createElement("tr");
  (headers || []).forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);

  // Rows are paged and searched on the backend for the full loaded dataset.
  tbody.innerHTML = "";
  (rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    headers.forEach((h) => {
      const td = document.createElement("td");
      const val = r[h];
      if (val === null || val === undefined) {
        td.innerHTML = `<span style="color:var(--accent-rose); opacity:0.8;">null</span>`;
      } else {
        td.textContent = String(val);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  const pageInfo = document.getElementById("gridPageInfo");
  state.gridFilteredTotal = total;
  if (pageInfo) pageInfo.textContent = total
    ? `Showing ${offset + 1}–${offset + (rows || []).length} of ${total.toLocaleString()} matching rows`
    : "No rows match this search.";
  const prev = document.getElementById("btnGridPrev");
  const next = document.getElementById("btnGridNext");
  if (prev) prev.disabled = state.gridPage <= 1;
  if (next) next.disabled = offset + (rows || []).length >= total;
}

async function loadPreviewPage(page = 1) {
  if (!state.currentDataset) return;
  state.gridPage = Math.max(1, page);
  if (!state.isBackendConnected || !state.activeDatasetId) {
    renderDataGrid(state.currentDataset.preview_rows, state.currentDataset.headers, state.currentDataset.preview_rows?.length || 0, 0);
    return;
  }
  const offset = (state.gridPage - 1) * state.gridPageSize;
  const params = new URLSearchParams({
    offset: String(offset), limit: String(state.gridPageSize), search: state.gridSearch || "",
  });
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.activeDatasetId)}/rows?${params}`);
    if (!res.ok) throw new Error(await responseError(res));
    const result = await res.json();
    renderDataGrid(result.rows, result.columns, result.total, result.offset);
  } catch (err) {
    showToast(err.message || "Could not load this page of rows.", "error");
  }
}

// =============================================================================
// 6. PAGE 2: SELECTION & INSPECTION
// =============================================================================

function populateSelectionControls() {
  const d = state.currentDataset;
  if (!d) return;

  // Populate column checklist
  const container = document.getElementById("selectColumnsList");
  if (container) {
    container.innerHTML = "";
    d.headers.forEach((h) => {
      const label = document.createElement("label");
      label.className = "pill-checkbox";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "selected_col";
      checkbox.value = h;
      checkbox.checked = true;
      const labelText = document.createElement("span");
      labelText.textContent = h;
      label.append(checkbox, labelText);
      container.appendChild(label);
    });
  }

  // Populate filter condition column dropdown
  const filterCol = document.getElementById("filterColSelect");
  if (filterCol) {
    filterCol.innerHTML = "";
    d.headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      filterCol.appendChild(opt);
    });
  }

  // Populate sort column dropdown
  const sortCol = document.getElementById("sortColSelect");
  if (sortCol) {
    sortCol.innerHTML = "";
    d.headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      sortCol.appendChild(opt);
    });
  }
}

function populateAdvancedControls() {
  const d = state.currentDataset;
  if (!d) return;
  const numericColumns = d.column_types?.numerical || [];
  const fill = (id, columns) => {
    const select = document.getElementById(id);
    if (!select) return;
    select.replaceChildren();
    columns.forEach((column) => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      select.appendChild(option);
    });
  };
  fill("insightColumnSelect", d.headers);
  fill("countConditionColumn", d.headers);
  fill("aggregateGroupSelect", d.headers);
  fill("aggregateValueSelect", numericColumns.length ? numericColumns : d.headers);
  fill("pivotIndexSelect", d.headers);
  fill("pivotColumnSelect", d.headers);
  fill("pivotValueSelect", numericColumns.length ? numericColumns : d.headers);
}

// =============================================================================
// 7. PAGE 3: PREPROCESSING & CLEANING
// =============================================================================

function populatePreprocessingControls() {
  const d = state.currentDataset;
  if (!d) return;

  const imputeCol = document.getElementById("imputeColSelect");
  if (imputeCol) {
    imputeCol.innerHTML = "";
    // Only columns with missing values or all columns
    d.headers.forEach((h) => {
      const colMeta = d.columns.find((c) => c.name === h);
      const missing = colMeta ? colMeta.missing : 0;
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = `${h} (${missing} missing)`;
      imputeCol.appendChild(opt);
    });
  }

  const outlierCol = document.getElementById("outlierColSelect");
  if (outlierCol) {
    outlierCol.innerHTML = "";
    const numCols = d.column_types?.numerical || [];
    numCols.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      outlierCol.appendChild(opt);
    });
  }

  const dropCol = document.getElementById("dropColSelect");
  if (dropCol) {
    dropCol.innerHTML = "";
    d.headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      dropCol.appendChild(opt);
    });
  }

  const textCol = document.getElementById("textColSelect");
  if (textCol) {
    textCol.innerHTML = "";
    d.headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      textCol.appendChild(opt);
    });
  }
}

// =============================================================================
// 8. PAGE 4: TRANSFORMATION & FEATURE ENGINEERING
// =============================================================================

function populateTransformationControls() {
  const d = state.currentDataset;
  if (!d) return;

  const scaleCol = document.getElementById("scaleColSelect");
  if (scaleCol) {
    scaleCol.innerHTML = "";
    const numCols = d.column_types?.numerical || [];
    numCols.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      scaleCol.appendChild(opt);
    });
  }

  const encodeCol = document.getElementById("encodeColSelect");
  if (encodeCol) {
    encodeCol.innerHTML = "";
    const catCols = d.column_types?.categorical || [];
    catCols.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      encodeCol.appendChild(opt);
    });
  }

  const dateCol = document.getElementById("dateColSelect");
  if (dateCol) {
    dateCol.innerHTML = "";
    (d.column_types?.datetime || []).forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      dateCol.appendChild(opt);
    });
    if (!dateCol.options.length) {
      dateCol.disabled = true;
      const opt = document.createElement("option");
      opt.textContent = "No date columns detected";
      dateCol.appendChild(opt);
    } else dateCol.disabled = false;
  }

  const numericColumns = d.column_types?.numerical || [];
  const derivedColumns = numericColumns.length ? numericColumns : d.headers;
  ["derivedLeftSelect", "derivedRightSelect"].forEach((id) => {
    const select = document.getElementById(id);
    if (!select) return;
    select.replaceChildren();
    derivedColumns.forEach((column) => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      select.appendChild(option);
    });
  });
  const conditionColumn = document.getElementById("conditionalColumnSelect");
  if (conditionColumn) {
    conditionColumn.replaceChildren();
    d.headers.forEach((column) => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      conditionColumn.appendChild(option);
    });
  }
  const availableColumns = new Set(d.headers.map((column) => String(column).toLowerCase()));
  const domainFeatures = [
    ["btnCreateFamilySize", ["Name", "SibSp", "Parch"]],
    ["btnCreateAgeGroup", ["Name", "Age"]],
    ["btnCreateIsAlone", ["Name", "SibSp", "Parch"]],
    ["btnCreateFareCategory", ["Name", "Fare"]],
    ["btnCreateCabinMissing", ["Name", "Cabin"]],
    ["btnExtractTicketFeature", ["Name", "Ticket"]],
  ];
  domainFeatures.forEach(([id, required]) => {
    const button = document.getElementById(id);
    if (!button) return;
    const missingColumns = required.filter((column) => !availableColumns.has(column.toLowerCase()));
    button.disabled = missingColumns.length > 0;
    button.title = missingColumns.length ? `Requires columns: ${missingColumns.join(", ")}` : "Available for this dataset";
  });
}

// =============================================================================
// 9. PAGE 5: MACHINE LEARNING & MINING
// =============================================================================

function populateMiningControls() {
  const d = state.currentDataset;
  if (!d) return;

  // Target label for Classification
  const classTarget = document.getElementById("mlClassTarget");
  if (classTarget) {
    classTarget.innerHTML = "";
    const preferredTarget = d.headers.find((name) => ["survived", "target", "label", "churn", "department"].includes(name.toLowerCase()))
      || d.headers.find((name) => !(d.identifier_columns || []).includes(name))
      || d.headers[0];
    d.headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      if (h === preferredTarget) opt.selected = true;
      classTarget.appendChild(opt);
    });
  }

  // Regression Target
  const regTarget = document.getElementById("mlRegTarget");
  if (regTarget) {
    regTarget.innerHTML = "";
    const numCols = (d.column_types?.numerical || []).filter((name) => !(d.identifier_columns || []).includes(name));
    const candidates = numCols.length ? numCols : (d.column_types?.numerical || []);
    candidates.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      regTarget.appendChild(opt);
    });
    const preferredTarget = candidates.find((name) => ["salary", "fare", "revenue", "price"].includes(name.toLowerCase())) || candidates[candidates.length - 1];
    if (preferredTarget) regTarget.value = preferredTarget;
  }
}

// =============================================================================
// 10. PAGE 6: VISUALIZATION STUDIO
// =============================================================================

function populateVisualizationControls() {
  const d = state.currentDataset;
  if (!d) return;

  const chartX = document.getElementById("chartXSelect");
  const chartY = document.getElementById("chartYSelect");
  if (chartX && chartY) {
    chartX.innerHTML = "";
    chartY.innerHTML = "";

    d.headers.forEach((h) => {
      const optX = document.createElement("option");
      optX.value = h;
      optX.textContent = h;
      chartX.appendChild(optX);

      const optY = document.createElement("option");
      optY.value = h;
      optY.textContent = h;
      chartY.appendChild(optY);
    });

    const defaultCategory = d.column_types?.categorical?.[0] || d.headers[0];
    const defaultNumber = d.column_types?.numerical?.find((name) => !(d.identifier_columns || []).includes(name))
      || d.column_types?.numerical?.[0] || d.headers[1] || d.headers[0];
    chartX.value = defaultCategory;
    chartY.value = defaultNumber;
  }

}

// =============================================================================
// 11. PAGE 7: AI CHAT & INTENT NLP LAB
// =============================================================================

function populateAIPrompts() {
  const promptContainer = document.getElementById("aiQuickPrompts");
  if (!promptContainer) return;

  promptContainer.innerHTML = "";
  const d = state.currentDataset;
  const numericColumn = d?.column_types?.numerical?.[0] || d?.headers?.[0] || "a numeric column";
  const secondNumeric = d?.column_types?.numerical?.[1] || numericColumn;
  const categoryColumn = d?.column_types?.categorical?.[0] || d?.headers?.[1] || "a category";
  const prompts = [
    "Summarize this dataset",
    "Show the full data quality report",
    "List missing values by column",
    "Show descriptive statistics for numeric columns",
    "Check for duplicate rows",
    `Show the distribution of ${categoryColumn}`,
    `What is the average ${numericColumn} by ${categoryColumn}?`,
    `Show the strongest correlation between ${numericColumn} and ${secondNumeric}`,
    `Detect outliers in ${numericColumn}`,
    `Fill missing values in ${numericColumn} using the median`,
    `Normalize ${numericColumn} using Min-Max scaling`,
    `Show a histogram of ${numericColumn}`,
    "Cluster this dataset into 3 groups using K-Means",
    "Compare classification models using the detected target",
    "Show a correlation heatmap",
    "Recommend data cleaning steps",
    "Create a complete KDD findings report",
  ];

  prompts.forEach((p) => {
    const chip = document.createElement("div");
    chip.className = "quick-prompt-chip";
    chip.textContent = p;
    chip.addEventListener("click", () => {
      const input = document.getElementById("aiChatInput");
      if (input) {
        input.value = p;
        sendAIMessage(p);
      }
    });
    promptContainer.appendChild(chip);
  });
}

// =============================================================================
// 12. OLLAMA-POWERED UNIVERSAL OPERATION DISPATCHER
//     All page buttons generate a natural-language question and route it
//     through Ollama (/ask) → plan_query() → execute_plan(). No if-else task
//     routing exists in the frontend — Ollama decides the correct function.
// =============================================================================

function renderSimpleTable(tableId, columns, rows) {
  const table = document.getElementById(tableId);
  if (!table) return;
  let thead = table.querySelector("thead");
  let tbody = table.querySelector("tbody");
  if (!thead) { thead = document.createElement("thead"); table.appendChild(thead); }
  if (!tbody) { tbody = document.createElement("tbody"); table.appendChild(tbody); }
  thead.innerHTML = "";
  const trH = document.createElement("tr");
  (columns || []).forEach((c) => { const th = document.createElement("th"); th.textContent = c; trH.appendChild(th); });
  thead.appendChild(trH);
  tbody.innerHTML = "";
  (rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => { const td = document.createElement("td"); td.textContent = String(r[c] ?? "-"); tr.appendChild(td); });
    tbody.appendChild(tr);
  });
}

/**
 * Core dispatcher: sends a natural-language question to Ollama via /ask.
 * Ollama's plan_query() figures out the right task; execute_plan() runs it.
 *
 * @param {string}  question         – Natural-language question for Ollama
 * @param {object}  outputOptions    – { tableId, chartContainerId, messageId }
 * @param {boolean} addToChatHistory – Echo Q&A in the AI chat panel
 */
async function askLLM(question, outputOptions = {}, addToChatHistory = false) {
  if (!question || !question.trim()) return;

  const d = state.currentDataset;
  if (!d) { showToast("Load a dataset before running operations.", "error"); return; }
  if (!state.isBackendConnected || !state.activeDatasetId) {
    showToast("Connect to app.py and load a dataset before running operations.", "error");
    return;
  }

  if (outputOptions.messageId) {
    const msgEl = document.getElementById(outputOptions.messageId);
    if (msgEl) msgEl.textContent = "⏳ Ollama is planning the operation…";
  }
  if (addToChatHistory) {
    appendChatBubble("user", question);
    const input = document.getElementById("aiChatInput");
    if (input) input.value = "";
  }

  showToast(`Asking Ollama: "${question.slice(0, 55)}…"`, "info");

  try {
    const selectedColumns = Array.from(
      document.querySelectorAll("#selectColumnsList input[name='selected_col']:checked")
    ).map((el) => el.value);

    const res = await fetch(`/api/datasets/${encodeURIComponent(state.activeDatasetId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, columns: selectedColumns }),
    });
    if (!res.ok) throw new Error(await responseError(res));
    const payload = await res.json();

    updatePlanInspector(payload);
    if (payload.intent) _applyIntentDisplay(payload);

    if (outputOptions.tableId && payload.table) {
      renderSimpleTable(outputOptions.tableId, payload.table.columns, payload.table.rows);
    }
    if (outputOptions.chartContainerId) {
      const container = document.getElementById(outputOptions.chartContainerId);
      if (container) {
        container.innerHTML = payload.chart
          ? `<img src="${payload.chart}" class="chart-matplotlib-img" alt="Ollama-generated visualization" />`
          : '<span style="color:var(--text-dim);font-size:12px;">No chart was generated for this operation.</span>';
      }
    }

    const answer = payload.answer || payload.message || "Operation completed.";
    if (outputOptions.messageId) {
      const msgEl = document.getElementById(outputOptions.messageId);
      if (msgEl) msgEl.textContent = answer;
    }

    if (payload.dataset) setDatasetState(payload.dataset);

    if (addToChatHistory) {
      const bubble = appendChatBubble("assistant", answer);
      if (payload.table) renderChatTable(bubble, payload.table);
      if (payload.chart) renderChatChart(bubble, payload.chart);
      // Also mirror into the result panel below the chat
      _renderAgentResultPanel(payload);
    } else {
      showToast(answer.length > 120 ? answer.slice(0, 120) + "…" : answer, "success");
    }

  } catch (err) {
    const message = err.message || "Ollama request failed.";
    if (outputOptions.messageId) {
      const msgEl = document.getElementById(outputOptions.messageId);
      if (msgEl) msgEl.textContent = `❌ ${message}`;
    }
    if (addToChatHistory) appendChatBubble("assistant", `Could not complete: ${message}`);
    showToast(message, "error");
  }
}

function _applyIntentDisplay(payload) {
  const badge = document.getElementById("intentPredictedBadge");
  const meter = document.getElementById("intentConfidenceMeter");
  if (badge && payload.intent) badge.textContent = payload.intent;
  if (meter && payload.confidence != null) meter.textContent = `${Math.round(payload.confidence * 100)}%`;
}

function _renderAgentResultPanel(payload) {
  const msg   = document.getElementById("aiResultMessage");
  const chart = document.getElementById("aiResultChart");
  if (msg)   msg.textContent = payload.answer || payload.message || "Analysis completed.";
  if (chart) {
    chart.replaceChildren();
    if (payload.chart) {
      const img = document.createElement("img");
      img.className = "chart-matplotlib-img";
      img.alt = "Visualization from dataset";
      img.src = payload.chart;
      chart.appendChild(img);
    } else {
      chart.textContent = "No visualization was returned for this question.";
    }
  }
  if (payload.table) renderSimpleTable("aiResultTable", payload.table.columns, payload.table.rows);
}

// =============================================================================
// 13. VISUALIZATION STUDIO — Ollama decides chart task from NL description
// =============================================================================

async function generateStudioChart() {
  const chartType = document.getElementById("chartTypeSelect")?.value || "bar";
  const xCol      = document.getElementById("chartXSelect")?.value;
  const yCol      = document.getElementById("chartYSelect")?.value;
  const questions = {
    bar:       `Show a bar chart of ${yCol} grouped by ${xCol}`,
    pie:       `Show a pie chart of the distribution of ${xCol}`,
    histogram: `Show a histogram of the distribution of ${xCol}`,
    scatter:   `Show a scatter plot of ${xCol} versus ${yCol}`,
    line:      `Show a line trend chart of ${yCol} over ${xCol}`,
    box:       `Show a box plot of ${yCol} grouped by ${xCol}`,
    heatmap:   "Show a correlation heatmap of all numeric columns",
  };
  const question = questions[chartType] || `Show a ${chartType} chart of ${xCol} and ${yCol}`;
  await askLLM(question, {
    tableId: "studioChartTable",
    chartContainerId: "studioChartResult",
    messageId: "studioChartMsg",
  });
}

// =============================================================================
// 14. AI CHAT — Routes through the same Ollama /ask pipeline
// =============================================================================

async function sendAIMessage(question) {
  if (!question || !question.trim()) return;
  if (!state.currentDataset || !state.isBackendConnected || !state.activeDatasetId) {
    appendChatBubble("assistant", "Connect to app.py and load a dataset before asking questions.");
    showToast("The analysis agent needs an active app.py dataset session.", "error");
    return;
  }
  // Route through unified Ollama dispatcher; echo in chat history
  await askLLM(question, {
    messageId: "aiResultMessage",
    chartContainerId: "aiResultChart",
    tableId: "aiResultTable",
  }, true);
  classifyIntentSentence(question);
}

function appendChatBubble(role, content) {
  const container = document.getElementById("aiChatHistory");
  if (!container) return null;
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble-${role}`;
  bubble.textContent = content;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

function renderChatTable(bubble, tableData) {
  if (!tableData?.columns?.length || !bubble) return;
  const wrapper = document.createElement("div");
  wrapper.className = "chat-result-table table-responsive";
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  tableData.columns.forEach((c) => { const th = document.createElement("th"); th.textContent = c; headRow.appendChild(th); });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  (tableData.rows || []).slice(0, 12).forEach((row) => {
    const tr = document.createElement("tr");
    tableData.columns.forEach((c) => { const td = document.createElement("td"); td.textContent = row[c] == null ? "—" : String(row[c]); tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrapper.appendChild(table);
  bubble.appendChild(wrapper);
}

function renderChatChart(bubble, chartUrl) {
  if (!bubble) return;
  const img = document.createElement("img");
  img.className = "chat-result-chart";
  img.alt = "Generated dataset visualization";
  img.src = chartUrl;
  bubble.appendChild(img);
}

function updatePlanInspector(payload) {
  const planBox = document.getElementById("aiPlanInspector");
  if (!planBox) return;
  planBox.textContent = JSON.stringify({
    detected_intent: payload.intent || "analytical_query",
    ml_confidence: payload.confidence ?? null,
    execution_architecture: "Ollama → plan_query() → execute_plan() [Zero-Eval Pandas]",
    synthesized_plan: payload.plan || { task: "plan unavailable", status: "completed" },
  }, null, 2);
}

async function classifyIntentSentence(sentence) {
  if (!sentence || !sentence.trim()) return;
  const resultBadge = document.getElementById("intentPredictedBadge");
  const meter       = document.getElementById("intentConfidenceMeter");
  const probList    = document.getElementById("intentProbList");
  try {
    const res = await fetch("/api/intent/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: sentence }),
    });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    if (resultBadge) resultBadge.textContent = data.predicted_intent;
    if (meter) meter.textContent = `${data.confidence_percentage}%`;
    if (probList && data.distribution) {
      probList.replaceChildren();
      data.distribution.slice(0, 5).forEach((item) => {
        const row  = document.createElement("div");
        row.className = "intent-probability-row";
        const lbl  = document.createElement("span"); lbl.textContent = item.intent;
        const val  = document.createElement("span"); val.textContent = `${item.percentage}%`;
        const line = document.createElement("div");  line.className = "intent-probability-track";
        const fill = document.createElement("div");  fill.className = "intent-probability-fill";
        fill.style.width = `${Math.max(0, Math.min(100, Number(item.percentage) || 0))}%`;
        line.appendChild(fill);
        row.append(lbl, val, line);
        probList.appendChild(row);
      });
    }
  } catch (err) {
    if (resultBadge) resultBadge.textContent = "Unavailable";
    if (meter) meter.textContent = "—";
    showToast(err.message || "Could not classify this intent.", "error");
  }
}

async function loadIntentMetrics() {
  try {
    const res = await fetch("/api/intent/metrics");
    if (res.ok) {
      const data = await res.json();
      const el   = document.getElementById("intentModelMetrics");
      if (el) el.innerHTML = `Accuracy: <b>${data.accuracy_percentage}%</b> • Macro F1: <b>${data.macro_f1}</b> • Classes: <b>${data.total_classes}</b> (Stratified 80/20)`;
    }
  } catch (_) {}
}

// =============================================================================
// 15. EVENT LISTENERS — ALL OPERATIONS ROUTED THROUGH OLLAMA /ask
// =============================================================================

function setupEventListeners() {
  // ── Theme & Navigation ──────────────────────────────────────────────────────
  document.getElementById("btnThemeToggle")?.addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
  document.getElementById("btnMobileNav")?.addEventListener("click", () =>
    document.querySelector(".app-sidebar")?.classList.toggle("open"));
  document.getElementById("btnAgentLauncher")?.addEventListener("click", () => switchPage("page-ai"));
  window.addEventListener("hashchange", () => {
    const p = window.location.hash.slice(1);
    if (p && document.getElementById(p)) switchPage(p, false);
  });

  // ── File Upload ─────────────────────────────────────────────────────────────
  document.getElementById("datasetFileInput")?.addEventListener("change", (e) => uploadDataset(e.target.files?.[0]));
  const dz = document.getElementById("datasetDropzone");
  dz?.addEventListener("dragover",  (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz?.addEventListener("dragleave", ()  => dz.classList.remove("dragover"));
  dz?.addEventListener("drop",      (e) => { e.preventDefault(); dz.classList.remove("dragover"); uploadDataset(e.dataTransfer?.files?.[0]); });

  // ── Preset Datasets ─────────────────────────────────────────────────────────
  document.getElementById("btnPresetTitanic")?.addEventListener("click", () => loadPresetDataset("titanic"));
  document.getElementById("btnPresetEmployee")?.addEventListener("click", () => loadPresetDataset("employee"));

  // ── Reset & Export ──────────────────────────────────────────────────────────
  document.getElementById("btnResetDataset")?.addEventListener("click", async () => {
    if (!state.activeDatasetId || !state.isBackendConnected) return showToast("Load a dataset before resetting it.", "error");
    try {
      const res = await fetch(`/api/datasets/${encodeURIComponent(state.activeDatasetId)}/reset`, { method: "POST" });
      if (!res.ok) throw new Error(await responseError(res));
      setDatasetState(await res.json());
      showToast("Dataset restored to its original uploaded state.", "success");
    } catch (err) { showToast(err.message || "Could not reset the dataset.", "error"); }
  });
  document.getElementById("btnExportDataset")?.addEventListener("click", () => {
    if (!state.isBackendConnected || !state.activeDatasetId) return showToast("Load a dataset before exporting.", "error");
    window.location.href = `/api/datasets/${encodeURIComponent(state.activeDatasetId)}/export`;
  });

  // ── Data Grid Pagination ────────────────────────────────────────────────────
  document.getElementById("gridSearchInput")?.addEventListener("input", (e) => {
    state.gridSearch = e.target.value; state.gridPage = 1;
    clearTimeout(state.gridSearchTimer);
    state.gridSearchTimer = setTimeout(() => loadPreviewPage(1), 250);
  });
  document.getElementById("gridPageSizeSelect")?.addEventListener("change", (e) => {
    state.gridPageSize = Number(e.target.value) || 10; loadPreviewPage(1);
  });
  document.getElementById("btnGridPrev")?.addEventListener("click", () => loadPreviewPage(state.gridPage - 1));
  document.getElementById("btnGridNext")?.addEventListener("click", () => loadPreviewPage(state.gridPage + 1));

  // ══════════════════════════════════════════════════════════════════════════════
  // STAGE 1: SELECTION & FILTER
  // ══════════════════════════════════════════════════════════════════════════════
  const selOut = { tableId: "selectionResultTable", messageId: "selectionResultMsg" };

  document.getElementById("btnRunSelectCols")?.addEventListener("click", () => {
    const selected = Array.from(document.querySelectorAll("input[name='selected_col']:checked")).map(el => el.value);
    if (!selected.length) return showToast("Check at least one column to select.", "error");
    askLLM(`Show only these columns: ${selected.join(", ")}`, selOut);
  });
  document.getElementById("btnRunSample")?.addEventListener("click", () => {
    const n = parseInt(document.getElementById("sampleRowsInput")?.value || 5);
    askLLM(`Show a random sample of ${n} rows`, selOut);
  });
  document.getElementById("btnRunSlice")?.addEventListener("click", () => {
    const start = Number(document.getElementById("sliceStartInput")?.value || 0);
    const end   = Number(document.getElementById("sliceEndInput")?.value   || 20);
    askLLM(`Show rows from index ${start} to ${end}`, selOut);
  });
  document.getElementById("btnRunTopN")?.addEventListener("click", () => {
    const col = document.getElementById("sortColSelect")?.value;
    const n   = parseInt(document.getElementById("topNCountInput")?.value || 5);
    askLLM(`Show the top ${n} rows by ${col}`, selOut);
  });
  document.getElementById("btnRunBottomN")?.addEventListener("click", () => {
    const col = document.getElementById("sortColSelect")?.value;
    const n   = parseInt(document.getElementById("topNCountInput")?.value || 5);
    askLLM(`Show the bottom ${n} rows by ${col}`, selOut);
  });
  document.getElementById("btnRunSort")?.addEventListener("click", () => {
    const col = document.getElementById("sortColSelect")?.value;
    const asc = document.getElementById("sortDirectionSelect")?.value === "true";
    askLLM(`Sort all rows by ${col} in ${asc ? "ascending" : "descending"} order`, selOut);
  });
  document.getElementById("btnRunFilter")?.addEventListener("click", () => {
    const col   = document.getElementById("filterColSelect")?.value;
    const op    = document.getElementById("filterOperatorSelect")?.value;
    const value = document.getElementById("filterValueInput")?.value;
    const opLbl = { "==": "equals", "!=": "does not equal", ">": "is greater than", ">=": "is at least",
                    "<": "is less than", "<=": "is at most", "contains": "contains", "isnull": "is missing" }[op] || op;
    askLLM(`Filter rows where ${col} ${opLbl} ${value}`, selOut);
  });
  document.getElementById("btnIdentifyIDs")?.addEventListener("click", () =>
    askLLM("Detect and identify the ID columns in this dataset", selOut));
  document.getElementById("btnIdentifyTarget")?.addEventListener("click", () =>
    askLLM("Detect the most likely target or label column in this dataset", selOut));
  document.getElementById("btnSelectByType")?.addEventListener("click", () => {
    const dtype = document.getElementById("dtypeSelect")?.value;
    askLLM(`Select all ${dtype} columns`, selOut);
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // EDA & AGGREGATION
  // ══════════════════════════════════════════════════════════════════════════════
  const insOut = { tableId: "insightResultTable", messageId: "insightResultMsg" };

  document.getElementById("btnRunProfile")?.addEventListener("click", () =>
    askLLM("Generate a full dataset profile report", insOut));
  document.getElementById("btnRunQualityReport")?.addEventListener("click", () =>
    askLLM("Generate a full data quality report", insOut));
  document.getElementById("btnRunEmptyStrings")?.addEventListener("click", () =>
    askLLM("Audit all empty or whitespace-only string values", insOut));
  document.getElementById("btnRunTypeConsistency")?.addEventListener("click", () =>
    askLLM("Check type consistency across all columns", insOut));
  document.getElementById("btnRunConstantColumns")?.addEventListener("click", () =>
    askLLM("Identify constant columns and likely ID columns", insOut));
  document.getElementById("btnRunQuartiles")?.addEventListener("click", () =>
    askLLM("Show a five-number summary with quartiles for all numeric columns", insOut));
  document.getElementById("btnRunStrongCorrelations")?.addEventListener("click", () =>
    askLLM("Find the strongest correlations between numeric columns", insOut));
  document.getElementById("btnCheckUniqueValues")?.addEventListener("click", () =>
    askLLM(`Check if ${document.getElementById("insightColumnSelect")?.value} contains unique values`, insOut));
  document.getElementById("btnCountUniqueValues")?.addEventListener("click", () =>
    askLLM(`Count the number of unique values in ${document.getElementById("insightColumnSelect")?.value}`, insOut));
  document.getElementById("btnShowValueCounts")?.addEventListener("click", () =>
    askLLM(`Show the frequency distribution of ${document.getElementById("insightColumnSelect")?.value}`, insOut));
  document.getElementById("btnCompareMeanMedian")?.addEventListener("click", () =>
    askLLM(`Compare the mean and median of ${document.getElementById("insightColumnSelect")?.value}`, insOut));
  document.getElementById("btnRunSkewness")?.addEventListener("click", () =>
    askLLM(`Calculate the skewness of ${document.getElementById("insightColumnSelect")?.value}`, insOut));
  document.getElementById("btnRunHighNullDrop")?.addEventListener("click", () => {
    const t = Number(document.getElementById("highNullThreshold")?.value || 50);
    askLLM(`Drop columns where more than ${t}% of values are missing`, insOut);
  });
  document.getElementById("btnRunGroupAggregate")?.addEventListener("click", () => {
    const grp  = document.getElementById("aggregateGroupSelect")?.value;
    const col  = document.getElementById("aggregateValueSelect")?.value;
    const meth = document.getElementById("aggregateMethodSelect")?.value;
    askLLM(`What is the ${meth} of ${col} grouped by ${grp}?`, insOut);
  });
  document.getElementById("btnRunPivot")?.addEventListener("click", () => {
    const idx  = document.getElementById("pivotIndexSelect")?.value;
    const cols = document.getElementById("pivotColumnSelect")?.value;
    const vals = document.getElementById("pivotValueSelect")?.value;
    const meth = document.getElementById("aggregateMethodSelect")?.value;
    askLLM(`Build a pivot table with ${idx} as index, ${cols} as columns, and ${meth} of ${vals} as values`, insOut);
  });
  document.getElementById("btnRunCrosstab")?.addEventListener("click", () => {
    const c1 = document.getElementById("pivotIndexSelect")?.value;
    const c2 = document.getElementById("pivotColumnSelect")?.value;
    if (!c1 || !c2 || c1 === c2) return showToast("Choose two different columns for the cross-tab.", "error");
    askLLM(`Run a cross-tabulation between ${c1} and ${c2}`, insOut);
  });
  document.getElementById("btnCountMatchingRows")?.addEventListener("click", () => {
    const col  = document.getElementById("countConditionColumn")?.value;
    const op   = document.getElementById("countConditionOperator")?.value;
    const val  = document.getElementById("countConditionValue")?.value;
    const opLbl = { "==": "equals", "!=": "does not equal", ">": "is greater than", ">=": "is at least",
                    "<": "is less than", "<=": "is at most", "contains": "contains", "isnull": "is missing" }[op] || op;
    askLLM(`Count rows where ${col} ${opLbl} ${val}`, insOut);
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // STAGE 2: PREPROCESSING & CLEANING
  // ══════════════════════════════════════════════════════════════════════════════
  const cleanOut = { tableId: "cleaningResultTable", messageId: "cleanAuditMsg" };

  document.getElementById("btnApplyImpute")?.addEventListener("click", () => {
    const col      = document.getElementById("imputeColSelect")?.value;
    const strategy = document.getElementById("imputeMethodSelect")?.value;
    const fillVal  = document.getElementById("imputeConstantValue")?.value;
    let q = `Fill missing values in ${col} using ${strategy}`;
    if (strategy === "constant" && fillVal) q += ` with the value ${fillVal}`;
    askLLM(q, cleanOut);
  });
  document.getElementById("btnDetectOutliers")?.addEventListener("click", () =>
    askLLM(`Detect outliers in ${document.getElementById("outlierColSelect")?.value} using the IQR method`, cleanOut));
  document.getElementById("btnCapOutliers")?.addEventListener("click", () =>
    askLLM(`Cap and Winsorize outliers in ${document.getElementById("outlierColSelect")?.value}`, cleanOut));
  document.getElementById("btnRemoveOutliers")?.addEventListener("click", () =>
    askLLM(`Remove rows that are outliers in ${document.getElementById("outlierColSelect")?.value}`, cleanOut));
  document.getElementById("btnDropDuplicates")?.addEventListener("click", () =>
    askLLM("Drop all duplicate rows from the dataset", cleanOut));
  document.getElementById("btnTrimWhitespace")?.addEventListener("click", () =>
    askLLM("Trim leading and trailing whitespace from all string columns", cleanOut));
  document.getElementById("btnDropMissing")?.addEventListener("click", () =>
    askLLM("Drop all rows that contain any missing values", cleanOut));
  document.getElementById("btnHandleInfinite")?.addEventListener("click", () =>
    askLLM("Replace all infinite values in numeric columns", cleanOut));
  document.getElementById("btnDropColumn")?.addEventListener("click", () => {
    const col = document.getElementById("dropColSelect")?.value;
    askLLM(`Drop the column named ${col} from the dataset`, cleanOut);
  });
  document.getElementById("btnCleaningSummary")?.addEventListener("click", () =>
    askLLM("Show a before and after cleaning summary of the dataset", cleanOut));
  document.getElementById("btnSetTextCase")?.addEventListener("click", () => {
    const col  = document.getElementById("textColSelect")?.value;
    const caseType = document.getElementById("textCaseSelect")?.value;
    askLLM(`Convert ${col} to ${caseType} case`, cleanOut);
  });
  document.getElementById("btnStandardizeCategories")?.addEventListener("click", () =>
    askLLM(`Standardize category values in ${document.getElementById("textColSelect")?.value}`, cleanOut));
  document.getElementById("btnConvertNumeric")?.addEventListener("click", () =>
    askLLM(`Convert ${document.getElementById("textColSelect")?.value} to numeric data type`, cleanOut));
  document.getElementById("btnConvertDate")?.addEventListener("click", () =>
    askLLM(`Convert ${document.getElementById("textColSelect")?.value} to datetime format`, cleanOut));

  // ══════════════════════════════════════════════════════════════════════════════
  // STAGE 3: TRANSFORMATION & FEATURE ENGINEERING
  // ══════════════════════════════════════════════════════════════════════════════
  const transOut = { tableId: "transResultTable", messageId: "transResultMsg" };

  document.getElementById("btnApplyScaling")?.addEventListener("click", () => {
    const col    = document.getElementById("scaleColSelect")?.value;
    const method = document.getElementById("scaleMethodSelect")?.value;
    const label  = { minmax: "Min-Max normalization to [0,1]", standard: "Z-score standardization", robust: "robust scaling (median/IQR)" }[method] || method;
    askLLM(`Apply ${label} to ${col}`, transOut);
  });
  document.getElementById("btnApplyLogTransform")?.addEventListener("click", () =>
    askLLM(`Apply log(1+x) transformation to ${document.getElementById("scaleColSelect")?.value}`, transOut));
  document.getElementById("btnApplyBinning")?.addEventListener("click", () => {
    const col = document.getElementById("encodeColSelect")?.value || document.getElementById("scaleColSelect")?.value;
    askLLM(`Discretize ${col} into 4 equal-width bins`, transOut);
  });
  document.getElementById("btnApplyOneHot")?.addEventListener("click", () =>
    askLLM(`One-hot encode the column ${document.getElementById("encodeColSelect")?.value}`, transOut));
  document.getElementById("btnApplyLabelEncode")?.addEventListener("click", () =>
    askLLM(`Label encode ${document.getElementById("encodeColSelect")?.value} into integers`, transOut));
  document.getElementById("btnExtractDatePart")?.addEventListener("click", () => {
    const col  = document.getElementById("dateColSelect")?.value;
    const part = document.getElementById("datePartSelect")?.value;
    askLLM(`Extract the ${part} from the date column ${col}`, transOut);
  });
  document.getElementById("btnCreateFamilySize")?.addEventListener("click", () =>
    askLLM("Create a Family_Size feature by adding SibSp and Parch columns plus 1", transOut));
  document.getElementById("btnCreateAgeGroup")?.addEventListener("click", () =>
    askLLM("Create an Age_Group categorical feature by binning the Age column", transOut));
  document.getElementById("btnCreateIsAlone")?.addEventListener("click", () =>
    askLLM("Create an IsAlone binary flag: 1 if SibSp plus Parch equals 0, else 0", transOut));
  document.getElementById("btnCreateFareCategory")?.addEventListener("click", () =>
    askLLM("Create a Fare_Category feature by binning the Fare column into fare tiers", transOut));
  document.getElementById("btnCreateCabinMissing")?.addEventListener("click", () =>
    askLLM("Create a Cabin_Missing binary flag indicating where Cabin values are missing", transOut));
  document.getElementById("btnExtractTicketFeature")?.addEventListener("click", () =>
    askLLM("Extract the ticket prefix from the Ticket column as a new categorical feature", transOut));
  document.getElementById("btnCreateDerivedColumn")?.addEventListener("click", () => {
    const c1     = document.getElementById("derivedLeftSelect")?.value;
    const c2     = document.getElementById("derivedRightSelect")?.value;
    const op     = document.getElementById("derivedArithmeticSelect")?.value;
    const newCol = document.getElementById("derivedNameInput")?.value.trim();
    if (!newCol) return showToast("Enter a name for the new column.", "error");
    const opLbl  = { "+": "adding", "-": "subtracting", "*": "multiplying", "/": "dividing" }[op] || op;
    askLLM(`Create a derived column called ${newCol} by ${opLbl} ${c1} and ${c2}`, transOut);
  });
  document.getElementById("btnCreateConditionalColumn")?.addEventListener("click", () => {
    const condCol = document.getElementById("conditionalColumnSelect")?.value;
    const op      = document.getElementById("conditionalOperatorSelect")?.value;
    const condVal = document.getElementById("conditionalValueInput")?.value;
    const newCol  = document.getElementById("conditionalNameInput")?.value.trim();
    const trueV   = document.getElementById("conditionalTrueInput")?.value;
    const falseV  = document.getElementById("conditionalFalseInput")?.value;
    if (!newCol || condVal === "") return showToast("Enter a result column name and comparison value.", "error");
    askLLM(`Create a conditional column called ${newCol}: set to ${trueV} where ${condCol} ${op} ${condVal}, otherwise ${falseV}`, transOut);
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // STAGE 4: MACHINE LEARNING & DATA MINING
  // ══════════════════════════════════════════════════════════════════════════════
  const mlOut = { tableId: "miningResultTable", chartContainerId: "miningChartPreview", messageId: "miningResultMsg" };

  document.getElementById("btnRunCorrelation")?.addEventListener("click", () =>
    askLLM("Compute the Pearson correlation matrix for all numeric columns", mlOut));
  document.getElementById("btnRunKMeans")?.addEventListener("click", () => {
    const k = parseInt(document.getElementById("kmeansClusterCount")?.value || 3);
    askLLM(`Cluster the dataset into ${k} groups using K-Means clustering`, mlOut);
  });
  document.getElementById("btnTrainClassification")?.addEventListener("click", () => {
    const target = document.getElementById("mlClassTarget")?.value || "Survived";
    const model  = document.getElementById("mlClassModelSelect")?.value || "decision_tree";
    const lbl    = { decision_tree: "Decision Tree", logistic_regression: "Logistic Regression", random_forest: "Random Forest" }[model] || model;
    askLLM(`Train a ${lbl} classifier to predict ${target} with an 80/20 train-test split and show evaluation metrics`, mlOut);
  });
  document.getElementById("btnRunPCA")?.addEventListener("click", () =>
    askLLM("Run PCA dimensionality reduction and show a 2D scatter projection", mlOut));
  document.getElementById("btnTrainRegression")?.addEventListener("click", () => {
    const target = document.getElementById("mlRegTarget")?.value;
    askLLM(`Train a linear regression model to predict ${target}`, mlOut);
  });
  document.getElementById("btnRunDBSCAN")?.addEventListener("click", () =>
    askLLM("Run DBSCAN density-based clustering on this dataset", mlOut));
  document.getElementById("btnRunModelComparison")?.addEventListener("click", () => {
    const target = document.getElementById("mlClassTarget")?.value;
    askLLM(`Compare multiple classification models for predicting ${target}`, mlOut);
  });
  document.getElementById("btnRunFeatureImportance")?.addEventListener("click", () => {
    const target = document.getElementById("mlClassTarget")?.value;
    askLLM(`Show feature importance scores for predicting ${target}`, mlOut);
  });
  document.getElementById("btnRunCrossValidation")?.addEventListener("click", () => {
    const target = document.getElementById("mlClassTarget")?.value;
    askLLM(`Run cross-validation for classifying ${target}`, mlOut);
  });

  // ── Stage 5: Visualization Studio ──────────────────────────────────────────
  document.getElementById("btnUpdateStudioChart")?.addEventListener("click", generateStudioChart);

  // ── AI Chat ─────────────────────────────────────────────────────────────────
  document.getElementById("btnSendAIMessage")?.addEventListener("click", () => {
    const input = document.getElementById("aiChatInput");
    if (input) sendAIMessage(input.value);
  });
  document.getElementById("aiChatInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendAIMessage(e.target.value);
  });

  // ── Intent Tester ─────────────────────────────────────────────────────────────
  document.getElementById("btnTestIntent")?.addEventListener("click", () => {
    const s = document.getElementById("intentTestInput")?.value;
    classifyIntentSentence(s);
  });
}

// =============================================================================
// 16. UTILITY FUNCTIONS
// =============================================================================

function formatNumber(val) {
  if (val === null || val === undefined) return "-";
  if (typeof val === "number") return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2);
  return escapeHTML(val);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const text = document.createElement("span");
  text.textContent = message;
  toast.appendChild(text);
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
