import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import git

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from codelens.chunker import chunk_repo
from codelens.embedder import embed_chunks, init_chromadb, load_model, store_chunks
from codelens.evaluator import QueryLogger, answer_relevance_score, log_query
from codelens.graph_builder import build_graph
from codelens.ingestion import accept_input, walk_files
from codelens.llm import ask
from codelens.retriever import HybridRetriever

load_dotenv()

# Inject GROQ_API_KEY from Streamlit secrets if available (Streamlit Cloud deployment)
try:
    _sk = st.secrets.get("GROQ_API_KEY", "")
    if _sk:
        os.environ["GROQ_API_KEY"] = _sk
except Exception:
    pass

st.set_page_config(
    page_title="CodeLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── LANGUAGE PALETTE (muted, non-neon) ──────────────────────────────────────
_LANG_COLORS: dict[str, str] = {
    "python":     "#3b82f6",
    "javascript": "#f59e0b",
    "typescript": "#8b5cf6",
    "go":         "#06b6d4",
    "java":       "#ef4444",
    "rust":       "#f97316",
    "cpp":        "#ec4899",
    "tsx":        "#a855f7",
}

def _lang_color(lang: str) -> str:
    return _LANG_COLORS.get(lang, "#6b7280")

# ─── LUCIDE SVG ICONS ────────────────────────────────────────────────────────
_LUCIDE_PATHS: dict[str, str] = {
    "git-branch":     '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
    "folder-open":    '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "layers":         '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "database":       '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "network":        '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
    "check-circle-2": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "arrow-up-circle":'<circle cx="12" cy="12" r="10"/><path d="m16 12-4-4-4 4"/><path d="M12 16V8"/>',
    "loader":         '<path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/>',
    "circle":         '<circle cx="12" cy="12" r="10"/>',
    "file-code":      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="m9 18 3-3-3-3"/><path d="m5 12-3 3 3 3"/>',
    "search":         '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
}

def _icon(name: str, size: int = 16, color: str = "currentColor", cls: str = "") -> str:
    """Render an inline Lucide SVG icon."""
    paths = _LUCIDE_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" class="lucide {cls}" '
        f'style="vertical-align:middle;flex-shrink:0">{paths}</svg>'
    )

def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts

# ─── DESIGN SYSTEM ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #0a0c10;
  --surface:    #0d1117;
  --surface-2:  #161b22;
  --surface-3:  #21262d;
  --border:     rgba(255,255,255,0.08);
  --border-2:   rgba(255,255,255,0.04);
  --text:       #e6edf3;
  --text-muted: #7d8590;
  --text-faint: #484f58;
  --accent:     #2dd4bf;
  --accent-hov: #14b8a6;
  --accent-dim: rgba(45,212,191,0.12);
  --red:        #f85149;
  --yellow:     #d29922;
  --green:      #3fb950;
  --radius:     6px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 28px;
}

/* ── GLOBAL ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body { background-color: var(--bg) !important; }
.stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.block-container {
  padding: 1.5rem 2.5rem 4rem !important;
  max-width: 1400px !important;
}
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── HIDE SIDEBAR ENTIRELY ───────────────────────────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 20px !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
  color: #e6edf3 !important;
  border-bottom: 2px solid #2dd4bf !important;
  background: var(--surface-2) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background-color: #2dd4bf !important;
}
.stTabs [data-baseweb="tab-border"] {
  background-color: var(--border) !important;
}
[data-testid="stTabPanel"] { background: transparent !important; padding-top: 1.75rem !important; }

/* ── METRIC CARDS ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-bottom: 2px solid var(--accent) !important;
  border-radius: var(--radius) !important;
  padding: 16px 18px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 28px !important;
  font-weight: 500 !important;
  color: var(--text) !important;
  line-height: 1.2 !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
  background: #2dd4bf !important;
  color: #0a0c10 !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  border: none !important;
  border-radius: 6px !important;
  height: 44px !important;
  padding: 0 22px !important;
  transition: background 150ms ease !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover,
.stButton > button[kind="primary"]:hover {
  background: #14b8a6 !important;
  color: #0a0c10 !important;
}
.stButton > button:disabled,
.stButton > button[kind="primary"]:disabled {
  background: var(--surface-3) !important;
  color: var(--text-faint) !important;
  cursor: not-allowed !important;
}

/* ── TEXT INPUT ──────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 14px !important;
  padding: 11px 14px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
  color: var(--text-faint) !important;
}
[data-testid="stTextInput"] label {
  color: var(--text-muted) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  margin-bottom: 6px !important;
}

/* ── MULTISELECT (teal pills) ────────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  min-height: 40px !important;
}
[data-baseweb="tag"] {
  background: rgba(45,212,191,0.12) !important;
  border: 1px solid rgba(45,212,191,0.3) !important;
  color: #2dd4bf !important;
  border-radius: 99px !important;
  font-size: 12px !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stMultiSelect"] label {
  color: var(--text-muted) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  margin-bottom: 6px !important;
}

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: #161b22 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 6px !important;
  overflow: hidden !important;
  margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--text-muted) !important;
  font-size: 12px !important;
  font-family: 'JetBrains Mono', monospace !important;
  padding: 10px 14px !important;
  background: var(--surface-3) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

/* ── CODE BLOCKS ─────────────────────────────────────────────────────────── */
[data-testid="stCode"] > div,
[data-testid="stCode"] pre {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
  color: var(--text) !important;
}

/* ── PROGRESS ────────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div {
  background: var(--surface-3) !important;
  border-radius: 2px !important;
  height: 3px !important;
}
[data-testid="stProgress"] > div > div {
  background: var(--accent) !important;
  border-radius: 2px !important;
}

/* ── STATUS / ALERTS ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  border-left: 3px solid var(--accent) !important;
  color: var(--text-muted) !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── TYPOGRAPHY ──────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}
h2 {
  font-size: 22px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  margin-bottom: 16px !important;
}
h3 { font-size: 16px !important; font-weight: 600 !important; }
h4 {
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--text-faint) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.5rem !important;
}
p {
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
  line-height: 1.7 !important;
}
hr {
  border: none !important;
  border-top: 1px solid rgba(255,255,255,0.08) !important;
  margin: var(--space-6) 0 !important;
}

/* ── SECTION HEADER (e.g. QUERY + SEMANTIC RAG) ──────────────────────────── */
.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  margin-top: 8px;
}
.section-title {
  font-size: 10px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.semantic-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid rgba(45,212,191,0.3);
  border-radius: 99px;
  font-size: 10px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ── ANSWER CONTAINER ────────────────────────────────────────────────────── */
.answer-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
  margin-bottom: 10px;
}
div:has(> [data-testid="stMarkdownContainer"] > .answer-header) + div [data-testid="stMarkdownContainer"] {
  background: #0d1117;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius);
  padding: 20px 22px;
  line-height: 1.8;
  font-size: 14px;
  color: var(--text);
}
div:has(> [data-testid="stMarkdownContainer"] > .answer-header) + div [data-testid="stMarkdownContainer"] code {
  background: var(--surface-3);
  color: var(--accent);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
}
div:has(> [data-testid="stMarkdownContainer"] > .answer-header) + div [data-testid="stMarkdownContainer"] pre {
  background: var(--bg) !important;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
}

/* ── SOURCE CARDS ────────────────────────────────────────────────────────── */
.source-card {
  padding: 14px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 8px;
}
.source-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.source-path {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--accent);
  word-break: break-all;
  flex: 1;
}
.source-lang-badge {
  font-size: 10px;
  padding: 2px 8px;
  background: var(--surface-3);
  border: 1px solid var(--border);
  border-radius: 99px;
  color: var(--text-muted);
  font-family: 'Inter', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  white-space: nowrap;
}
.relevance-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.relevance-label {
  font-size: 10px;
  color: var(--text-faint);
  font-family: 'Inter', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  white-space: nowrap;
}
.relevance-track {
  flex: 1;
  height: 4px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
}
.relevance-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
}
.relevance-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-muted);
  min-width: 44px;
  text-align: right;
}

/* ── STEP INDICATOR (replaces emoji loading) ─────────────────────────────── */
.step-list {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin: 12px 0;
}
.step-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 0;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
}
.step-row.completed { color: var(--text-muted); }
.step-row.active    { color: var(--text); font-weight: 500; }
.step-row.pending   { color: var(--text-faint); }
.step-icon-wrap {
  width: 18px; height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.step-detail {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-faint);
}
@keyframes spin { to { transform: rotate(360deg); } }
.lucide.spin { animation: spin 1s linear infinite; }

/* ── INLINE HINT (replaces emoji info box) ───────────────────────────────── */
.hint-box {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 13px;
  font-family: 'Inter', sans-serif;
}
.hint-box strong { color: var(--text); font-weight: 600; }

/* ── SUMMARY CARD (ingestion) ────────────────────────────────────────────── */
.summary-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px;
  margin: 20px 0;
}
.summary-row {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}
.summary-repo {
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  color: var(--accent);
  font-weight: 500;
}
.summary-branch {
  font-size: 11px;
  color: var(--text-faint);
  font-family: 'Inter', sans-serif;
  background: var(--surface-3);
  padding: 3px 8px;
  border-radius: 99px;
  border: 1px solid var(--border);
}
.summary-elapsed {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-faint);
}
.summary-stats {
  display: flex;
  gap: 40px;
  flex-wrap: wrap;
}
.summary-stat-label {
  font-size: 10px;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-family: 'Inter', sans-serif;
  margin-bottom: 4px;
}
.summary-stat-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 20px;
  color: var(--text);
}

/* ── LANGUAGE BAR ────────────────────────────────────────────────────────── */
.lang-bar-wrapper { margin-bottom: 20px; }
.lang-stacked-bar {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 14px;
  gap: 2px;
}
.lang-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}
.lang-legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  font-family: 'Inter', sans-serif;
}
.lang-legend-dot {
  width: 10px; height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}
.lang-pct {
  color: var(--text-faint);
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
}

/* ── FILE TABLE ──────────────────────────────────────────────────────────── */
.file-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.file-table th {
  padding: 10px 16px;
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  background: var(--surface-3);
  border-bottom: 1px solid var(--border);
}
.file-table th.num,
.file-table td.num { text-align: right; }
.file-table td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-2);
}
.file-table tr:nth-child(odd)  td { background: var(--surface); }
.file-table tr:nth-child(even) td { background: var(--surface-2); }
.file-table tr:hover td { background: var(--surface-3) !important; }
.file-path {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--accent);
}
.file-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--text-muted);
}
.lang-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 99px;
  font-size: 10px;
  font-family: 'Inter', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  background: var(--surface-3);
  border: 1px solid var(--border);
  color: var(--text-muted);
}

/* ── METRICS TABLE ───────────────────────────────────────────────────────── */
.metrics-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.metrics-table th {
  padding: 10px 14px;
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  background: var(--surface-3);
  border-bottom: 1px solid var(--border);
}
.metrics-table th.num,
.metrics-table td.num { text-align: right; }
.metrics-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  vertical-align: middle;
}
.metrics-table tr:hover td { background: var(--surface-3) !important; }
.metrics-table td.td-mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-muted);
}
.mini-rel {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.mini-rel-track {
  width: 70px;
  height: 4px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
}
.mini-rel-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
}
.mini-rel-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-muted);
  min-width: 36px;
}
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 99px;
  font-size: 10px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  white-space: nowrap;
}
.badge-green  { background: rgba(63,185,80,0.12);  color: var(--green);  border: 1px solid rgba(63,185,80,0.25); }
.badge-yellow { background: rgba(210,153,34,0.12); color: var(--yellow); border: 1px solid rgba(210,153,34,0.25); }
.badge-red    { background: rgba(248,81,73,0.12);  color: var(--red);    border: 1px solid rgba(248,81,73,0.25); }
.badge-muted  { background: var(--surface-3); color: var(--text-muted); border: 1px solid var(--border); }

/* ── STATUS BAR ──────────────────────────────────────────────────────────── */
.status-bar {
  position: fixed; bottom: 0; left: 0; right: 0; height: 22px;
  background: #007acc; color: #fff; padding: 0 14px;
  display: flex; align-items: center; gap: 16px;
  font-size: 11px; font-family: 'JetBrains Mono', monospace;
  z-index: 9999; letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ─── STEP INDICATOR ──────────────────────────────────────────────────────────
_PIPELINE_STEPS = [
    ("clone",   "Cloning Repository"),
    ("detect",  "Detecting Languages"),
    ("chunk",   "Chunking Code Files"),
    ("embed",   "Embedding Chunks"),
    ("graph",   "Building AST Graph"),
    ("done",    "Indexing Complete"),
]

def _render_step_list(placeholder, current_key: str, completed: set, details: dict | None = None) -> None:
    """Render the pipeline step list with Lucide icons."""
    details = details or {}
    rows = []
    for key, label in _PIPELINE_STEPS:
        if key in completed:
            state = "completed"
            icon_svg = _icon("check-circle-2", size=16, color="#2dd4bf")
        elif key == current_key:
            state = "active"
            icon_svg = _icon("loader", size=16, color="#2dd4bf", cls="spin")
        else:
            state = "pending"
            icon_svg = _icon("circle", size=16, color="#484f58")
        detail = details.get(key, "")
        detail_html = f'<span class="step-detail">{detail}</span>' if detail else ""
        rows.append(
            f'<div class="step-row {state}">'
            f'<span class="step-icon-wrap">{icon_svg}</span>'
            f'<span>{label}</span>'
            f'{detail_html}'
            f'</div>'
        )
    placeholder.markdown(
        f'<div class="step-list">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _render_lang_bar(languages: dict) -> None:
    total = sum(languages.values()) or 1
    segs, legend = "", ""
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        col = _lang_color(lang)
        segs += (
            f'<div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:2px" '
            f'title="{lang}: {pct:.1f}%"></div>'
        )
        legend += (
            f'<div class="lang-legend-item">'
            f'<div class="lang-legend-dot" style="background:{col}"></div>'
            f'<span>{lang}</span>'
            f'<span class="lang-pct">{count} ({pct:.1f}%)</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="lang-bar-wrapper">'
        f'<div class="lang-stacked-bar">{segs}</div>'
        f'<div class="lang-legend">{legend}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_file_table(file_data: dict) -> None:
    rows = "".join(
        f'<tr>'
        f'<td><span class="file-path">{src}</span></td>'
        f'<td><span class="lang-badge" style="border-left:2px solid {_lang_color(d["LANG"])}">'
        f'{d["LANG"]}</span></td>'
        f'<td class="num"><span class="file-num">{d["LOC"]:,}</span></td>'
        f'<td class="num"><span class="file-num">{d["CHUNKS"]}</span></td>'
        f'</tr>'
        for src, d in sorted(file_data.items())
    )
    st.markdown(
        f'<div style="overflow-x:auto">'
        f'<table class="file-table">'
        f'<thead><tr>'
        f'<th>Path</th><th>Language</th>'
        f'<th class="num">Lines</th><th class="num">Chunks</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def _render_query_history(entries: list) -> None:
    rows = ""
    for e in entries:
        ts   = _fmt_ts(e.get("timestamp", ""))
        qry_full = e.get("query", "").replace('"', "&quot;")
        qry  = qry_full[:60] + "…" if len(qry_full) > 60 else qry_full
        rel  = e.get("relevance_score", 0.0)
        lat  = e.get("latency_ms", 0.0)
        src  = e.get("sources_count", 0)
        rel_pct = min(100.0, float(rel) * 100) if isinstance(rel, (int, float)) else 0.0
        rel_s = f"{rel:.2f}" if isinstance(rel, float) else "—"
        lat_s = f"{lat/1000:.2f}s" if isinstance(lat, (int, float)) else "—"
        rows += (
            f"<tr>"
            f'<td title="{qry_full}">{qry}</td>'
            f"<td class='td-mono'>{ts}</td>"
            f"<td class='td-mono num'>{src}</td>"
            f"<td><div class='mini-rel'>"
            f"<div class='mini-rel-track'><div class='mini-rel-fill' style='width:{rel_pct:.1f}%'></div></div>"
            f"<span class='mini-rel-val'>{rel_s}</span>"
            f"</div></td>"
            f"<td class='td-mono num'>{lat_s}</td>"
            f"</tr>"
        )
    st.markdown(
        f'<div style="overflow-x:auto">'
        f'<table class="metrics-table">'
        f'<thead><tr>'
        f'<th>Query</th><th>Timestamp</th>'
        f'<th class="num">Sources</th><th>Relevance</th><th class="num">Latency</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


# ─── API KEY GUARD ───────────────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Add it to your environment variables "
        "or Streamlit secrets."
    )
    st.stop()

# ─── INPUT VALIDATION ────────────────────────────────────────────────────────
MAX_QUERIES_PER_MINUTE  = 10
MAX_INGESTIONS_PER_HOUR = 5
_REPO_SIZE_LIMIT_MB     = 500

_GITHUB_URL_RE = re.compile(r'^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$')
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your",
    "you are now",
    "act as",
    "jailbreak",
    "system prompt",
]
_DANGEROUS_PATTERNS = ["<script", "javascript:", "file://", "../", "..\\", "%2e%2e"]


def validate_github_url(url: str) -> tuple[bool, str]:
    url = url.strip()
    if not url:
        return False, "Please enter a GitHub URL."
    if not _GITHUB_URL_RE.match(url):
        return False, "Invalid URL format. Expected: https://github.com/owner/repository"
    for d in _DANGEROUS_PATTERNS:
        if d.lower() in url.lower():
            return False, "Invalid URL."
    return True, ""


def validate_query(query: str) -> tuple[bool, str]:
    query = query.strip()
    if not query:
        return False, "Please enter a question."
    if len(query) < 5:
        return False, "Query is too short. Please be more specific."
    if len(query) > 1000:
        return False, f"Query is too long ({len(query)} characters). Please keep it under 1000 characters."
    q_lower = query.lower()
    for phrase in _INJECTION_PHRASES:
        if phrase in q_lower:
            return False, "Query contains disallowed content. Please ask a question about the codebase."
    return True, ""


def _get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "repo_path":           None,
    "repo_name":           None,
    "file_list":           None,
    "chunks":              None,
    "graph":               None,
    "collection":          None,
    "stats":               None,
    "ingest_time_s":       None,
    "query_log":           [],
    "last_results":        [],
    "last_answer":         "",
    "active_tab":          0,
    "query_timestamps":    [],
    "ingest_timestamps":   [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_logger = QueryLogger()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["● ingestion.dash", "● query.console", "● metrics.live"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ingestion
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Repository Analysis")

    url_input = st.text_input(
        "GitHub URL or Local Path",
        placeholder="https://github.com/owner/repo  or  /path/to/local/repo",
        key="url_input",
    )

    analyze_clicked = st.button(
        "Analyze Repo", type="primary", disabled=not url_input
    )

    if analyze_clicked:
        st.session_state.chunks        = []
        st.session_state.graph         = None
        st.session_state.collection    = None
        st.session_state.repo_name     = ""
        st.session_state.stats         = None
        st.session_state.file_list     = None
        st.session_state.repo_path     = None
        st.session_state.last_results  = []
        st.session_state.last_answer   = ""
        st.session_state.ingest_time_s = None

        # ── Ingestion rate limit ───────────────────────────────────────────────
        _now = time.time()
        st.session_state.ingest_timestamps = [
            t for t in st.session_state.ingest_timestamps if _now - t < 3600
        ]
        if len(st.session_state.ingest_timestamps) >= MAX_INGESTIONS_PER_HOUR:
            st.error(
                "Ingestion limit reached. You can analyze up to 5 "
                "repositories per hour. Please wait before re-indexing."
            )
            st.stop()
        st.session_state.ingest_timestamps.append(_now)

        _url = url_input.strip()
        _is_github = "github.com" in _url
        _valid = True

        if _is_github:
            _ok, _msg = validate_github_url(_url)
            if not _ok:
                st.error(_msg)
                _valid = False

        if _valid:
            _t0 = time.perf_counter()
            _placeholder = st.empty()
            _completed: set = set()
            _details: dict[str, str] = {}
            _clone_path: str | None = None

            try:
                _render_step_list(_placeholder, "clone", _completed, _details)
                local_path, repo_name = accept_input(_url)
                if _is_github:
                    _clone_path = local_path
                _completed.add("clone")
                _details["clone"] = repo_name

                # ── Repo size guard ───────────────────────────────────────────
                if _is_github:
                    _repo_mb = _get_dir_size_mb(local_path)
                    if _repo_mb > _REPO_SIZE_LIMIT_MB:
                        shutil.rmtree(local_path, ignore_errors=True)
                        st.error(
                            f"Repository is too large ({_repo_mb:.0f} MB). "
                            f"CodeLens supports repositories up to {_REPO_SIZE_LIMIT_MB} MB."
                        )
                        st.stop()

                _render_step_list(_placeholder, "detect", _completed, _details)
                file_list, stats = walk_files(local_path)
                if not file_list:
                    raise ValueError("no_code_files")
                _completed.add("detect")
                _details["detect"] = f"{stats.total_files} files"

                _render_step_list(_placeholder, "chunk", _completed, _details)
                chunks = chunk_repo(local_path, file_list)
                _completed.add("chunk")
                _details["chunk"] = f"{len(chunks)} chunks"

                _render_step_list(_placeholder, "embed", _completed, _details)
                collection = init_chromadb(persist_dir=".chromadb", clear_existing=True)
                embedded   = embed_chunks(chunks)
                store_chunks(collection, embedded)
                _completed.add("embed")
                _details["embed"] = "ChromaDB"

                _render_step_list(_placeholder, "graph", _completed, _details)
                graph = build_graph(chunks)
                _completed.add("graph")
                _details["graph"] = f"{graph.number_of_nodes()} nodes · {graph.number_of_edges()} edges"

                _completed.add("done")
                _details["done"] = f"{stats.total_files} files · {len(chunks)} chunks"
                _render_step_list(_placeholder, "done", _completed, _details)

                if _clone_path:
                    shutil.rmtree(_clone_path, ignore_errors=True)

                st.session_state.repo_path     = local_path
                st.session_state.repo_name     = repo_name
                st.session_state.file_list     = file_list
                st.session_state.chunks        = chunks
                st.session_state.graph         = graph
                st.session_state.collection    = collection
                st.session_state.stats         = stats
                st.session_state.ingest_time_s = time.perf_counter() - _t0

            except git.exc.GitCommandError as exc:
                _msg = str(exc).lower()
                if "not found" in _msg or "does not exist" in _msg or "repository" in _msg:
                    st.error(
                        "Repository not found. Make sure the repo is public and the URL is correct."
                    )
                else:
                    st.error(
                        "Failed to clone repository. Check your internet connection and try again."
                    )
            except ValueError as exc:
                if str(exc) == "no_code_files":
                    st.error(
                        "This repository appears to have no supported code files to index."
                    )
                else:
                    st.error(f"Pipeline failed: {exc}")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    # ── Ingestion summary card + metrics ───────────────────────────────────────
    if st.session_state.chunks:
        _chunks  = st.session_state.chunks
        _stats   = st.session_state.stats
        _unique  = len({c.source for c in _chunks})
        _coverage = round(_unique / max(_stats.total_files, 1) * 100, 1)

        _elapsed = st.session_state.ingest_time_s
        _elapsed_str = f"{_elapsed:.1f}s" if _elapsed else "—"
        _total_langs = max(sum(_stats.languages.values()), 1)
        _lang_pcts = "  ·  ".join(
            f"{lang} {round(n / _total_langs * 100)}%"
            for lang, n in sorted(_stats.languages.items(), key=lambda x: -x[1])
        )

        st.markdown(
            f'<div class="summary-card">'
            f'<div class="summary-row">'
            f'<span class="summary-repo">◉ {st.session_state.repo_name}</span>'
            f'<span class="summary-branch">default branch</span>'
            f'<span class="summary-elapsed">{_elapsed_str}</span>'
            f'</div>'
            f'<div class="summary-stats">'
            f'<div><div class="summary-stat-label">Files</div>'
            f'<div class="summary-stat-value">{_stats.total_files}</div></div>'
            f'<div><div class="summary-stat-label">Chunks</div>'
            f'<div class="summary-stat-value">{len(_chunks)}</div></div>'
            f'<div><div class="summary-stat-label">Languages</div>'
            f'<div class="summary-stat-value" style="font-size:13px;color:var(--text-muted);padding-top:6px">{_lang_pcts}</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Files",    _stats.total_files)
        c2.metric("Embedded Chunks", len(_chunks))
        c3.metric("Lines of Code",  f"{_stats.total_lines:,}")
        c4.metric("Coverage",       f"{_coverage}%")

        st.markdown(
            '<div class="section-header" style="margin-top:24px">'
            '<span class="section-title">Language Breakdown</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _render_lang_bar(_stats.languages)

        st.markdown(
            '<div class="section-header" style="margin-top:20px">'
            '<span class="section-title">Indexed Files</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _root      = Path(st.session_state.repo_path)
        _file_data: dict[str, dict] = {}
        for _fp, _lang in (st.session_state.file_list or []):
            _rel = str(Path(_fp).relative_to(_root))
            _file_data[_rel] = {"LANG": _lang, "LOC": 0, "CHUNKS": 0}
        for chunk in _chunks:
            fd = _file_data.setdefault(
                chunk.source, {"LANG": chunk.language, "LOC": 0, "CHUNKS": 0}
            )
            fd["LANG"]   = chunk.language
            fd["CHUNKS"] += 1
            fd["LOC"]    = max(fd["LOC"], chunk.start_line + chunk.text.count("\n"))
        _render_file_table(_file_data)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Query Console
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── Top stats row ──────────────────────────────────────────────────────────
    _qs_stats = _logger.get_stats()
    _g = st.session_state.graph
    _s = st.session_state.stats
    qs = st.columns(5)
    qs[0].metric("Files Indexed",   _s.total_files if _s else 0)
    qs[1].metric("Chunks Embedded", len(st.session_state.chunks) if st.session_state.chunks else 0)
    qs[2].metric("Graph Nodes",     _g.number_of_nodes() if _g else 0)
    qs[3].metric("Graph Edges",     _g.number_of_edges() if _g else 0)
    qs[4].metric("Queries Logged",  _qs_stats["total_queries"])

    st.markdown("---")

    # ── Query input ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<span class="section-title">Query</span>'
        '<span class="semantic-badge">Semantic RAG</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Ask About the Codebase",
        placeholder="e.g. How does authentication work in this repo?",
        key="query_input",
    )

    _available_langs = (
        list(st.session_state.stats.languages.keys())
        if st.session_state.stats
        else ["python", "javascript", "typescript", "java", "cpp", "go", "tsx"]
    )
    lang_filter = st.multiselect(
        "Filter by Language",
        options=_available_langs,
        default=[],
        key="lang_filter",
    )

    _can_query = bool(query and st.session_state.collection)
    submit = st.button("Ask CodeLens", type="primary", disabled=not _can_query, width="stretch")

    if not st.session_state.collection:
        st.markdown(
            f'<div class="hint-box">'
            f'{_icon("arrow-up-circle", size=18, color="#2dd4bf")}'
            f'<span>Start by pasting a GitHub repo URL in the '
            f'<strong>ingestion.dash</strong> tab and clicking <strong>Analyze Repo</strong>.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if submit and _can_query:
        # ── Query rate limit ───────────────────────────────────────────────────
        _now = time.time()
        st.session_state.query_timestamps = [
            t for t in st.session_state.query_timestamps if _now - t < 60
        ]
        if len(st.session_state.query_timestamps) >= MAX_QUERIES_PER_MINUTE:
            st.error(
                "Rate limit reached. You can run up to 10 queries "
                "per minute. Please wait a moment and try again."
            )
            st.stop()
        st.session_state.query_timestamps.append(_now)

        # ── Query validation ───────────────────────────────────────────────────
        _q_ok, _q_msg = validate_query(query)
        if not _q_ok:
            st.error(_q_msg)
            st.stop()

        try:
            _model     = load_model()
            _retriever = HybridRetriever(
                collection=st.session_state.collection,
                graph=st.session_state.graph,
                model=_model,
            )
            t0 = time.perf_counter()

            with st.spinner("Retrieving relevant code..."):
                _results = _retriever.retrieve(query)
                if lang_filter:
                    _results = [r for r in _results if r.get("language", "") in lang_filter]
                _context = _retriever.get_context_string(_results)

            with st.spinner("Generating answer..."):
                _resp = ask(query, _context)

            _latency_ms = (time.perf_counter() - t0) * 1000
            _relevance  = answer_relevance_score(query, _resp["answer"])

            st.session_state.last_results = _results
            st.session_state.last_answer  = _resp["answer"]

            _entry = log_query(query, _resp["answer"], _results, _latency_ms, _relevance)
            _entry["repository"] = st.session_state.repo_name or ""
            _logger.log(_entry)

        except Exception as exc:
            _exc_str = str(exc).lower()
            if "api_key" in _exc_str or "authentication" in _exc_str or "unauthorized" in _exc_str:
                st.error("Query failed. Please check your GROQ_API_KEY and try again.")
            else:
                st.error(f"Query failed: {exc}")

    # ── Answer ─────────────────────────────────────────────────────────────────
    if st.session_state.last_answer:
        st.markdown(
            '<div class="answer-header">'
            '<span class="section-title">Answer</span>'
            '<span class="semantic-badge">Semantic RAG</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.last_answer)

        # ── Sources ───────────────────────────────────────────────────────────
        if st.session_state.last_results:
            st.markdown(
                '<div class="section-header" style="margin-top:24px">'
                '<span class="section-title">Sources</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            for _r in st.session_state.last_results:
                _score = _r["score"]
                _pct   = min(100.0, _score * 100)
                _lang  = _r.get("language", "text")
                _src   = _r.get("source", "")
                _start = 1
                _end   = _start + _r["text"].count("\n")

                st.markdown(f"""
                <div class="source-card">
                  <div class="source-meta">
                    <span class="source-path">{_src}</span>
                    <span class="source-lang-badge">{_lang}</span>
                  </div>
                  <div class="relevance-row">
                    <span class="relevance-label">Relevance</span>
                    <div class="relevance-track">
                      <div class="relevance-fill" style="width:{_pct:.1f}%"></div>
                    </div>
                    <span class="relevance-value">{_score:.3f}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"View Code  ·  {_src}  ·  L{_start}–{_end}", expanded=False):
                    st.code(_r["text"], language=_lang)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    _entries   = _logger.load()
    _all_stats = _logger.get_stats()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Queries",  _all_stats["total_queries"])
    k2.metric("Avg Relevance",  f"{_all_stats['avg_relevance']:.2f}")
    k3.metric("Avg Sources",    f"{_all_stats['avg_sources']:.1f}")
    k4.metric("Avg Latency",    f"{_all_stats['avg_latency_ms']/1000:.2f}s" if _all_stats['avg_latency_ms'] else "0.00s")

    if not _entries:
        st.markdown(
            f'<div class="hint-box" style="margin-top:24px">'
            f'{_icon("search", size=18, color="#2dd4bf")}'
            f'<span>No queries logged yet. Run your first query in the '
            f'<strong>query.console</strong> tab.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-header" style="margin-top:24px">'
            '<span class="section-title">Query Volume</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        _heat: dict[tuple[int, int], int] = defaultdict(int)
        for _e in _entries:
            try:
                _dt = datetime.fromisoformat(_e["timestamp"].replace("Z", "+00:00"))
                _heat[(_dt.weekday(), _dt.hour)] += 1
            except (ValueError, KeyError):
                pass

        _z = [[_heat.get((_day, _hr), 0) for _hr in range(24)] for _day in range(7)]
        _fig = go.Figure(go.Heatmap(
            z=_z, x=list(range(24)), y=_day_names,
            colorscale=[[0, "#161b22"], [1, "#2dd4bf"]],
            showscale=True,
        ))
        _fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Day of Week",
            height=280, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="#0a0c10", plot_bgcolor="#0d1117",
            font=dict(family="Inter, sans-serif", color="#7d8590", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(_fig, width="stretch")

        st.markdown(
            '<div class="section-header" style="margin-top:24px">'
            '<span class="section-title">Query History</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _sorted = sorted(_entries, key=lambda e: e.get("timestamp", ""), reverse=True)
        _render_query_history(_sorted)

# ─── STATUS BAR ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="status-bar">'
    '● main &nbsp;&nbsp; CodeLens &nbsp;&nbsp; RAG-Powered Code Intelligence &nbsp;&nbsp; ready'
    '</div>',
    unsafe_allow_html=True,
)
