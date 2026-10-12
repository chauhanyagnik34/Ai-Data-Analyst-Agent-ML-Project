"""
patch_appjs.py
--------------
Replaces sections 12–16 of app.js (the old executeTask if-else dispatcher and
all manual button wiring) with the new Ollama-routed askLLM() architecture.
Every button now generates a natural-language question sent to /ask, where
Ollama's plan_query() picks the right operation and execute_plan() runs it.
"""

import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.js")

# ── We keep everything up to (but not including) section 12 ──────────────────
KEEP_MARKER = (
    "// =============================================================================\n"
    "// 12. OPERATION EXECUTION DISPATCHER (LIVE API)"
)

NEW_TAIL = r'''// =============================================================================
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
'''

# ── Apply patch ───────────────────────────────────────────────────────────────
with open(SRC, "r", encoding="utf-8") as f:
    original = f.read()

idx = original.find(KEEP_MARKER)
if idx == -1:
    raise RuntimeError("Could not find section 12 marker in app.js")

kept_head = original[:idx]

with open(SRC, "w", encoding="utf-8") as f:
    f.write(kept_head + NEW_TAIL)

print(f"OK: app.js patched successfully.  Original kept up to char {idx}.")
print(f"    New file size: {len(kept_head + NEW_TAIL):,} chars")
