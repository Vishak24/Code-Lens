import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from codelens.chunker import chunk_repo
from codelens.embedder import embed_chunks, init_chromadb, load_model, store_chunks
from codelens.evaluator import QueryLogger, answer_relevance_score, log_query
from codelens.graph_builder import build_graph
from codelens.ingestion import accept_input, walk_files
from codelens.llm import ask, generate_followups
from codelens.retriever import HybridRetriever

load_dotenv()

st.set_page_config(
    page_title="CodeLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Base dark theme */
  .stApp { background-color: #1e1e1e; color: #d4d4d4; }
  section[data-testid="stSidebar"] { background-color: #252526 !important; border-right: 1px solid #3e3e3e; }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Metric cards */
  [data-testid="stMetric"] { background: #2d2d2d; border: 1px solid #3e3e3e; border-radius: 6px; padding: 16px !important; }
  [data-testid="stMetricValue"] { color: #4ec9b0 !important; font-size: 2rem !important; font-family: 'Courier New', monospace; }
  [data-testid="stMetricLabel"] { color: #858585 !important; font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }

  /* Buttons */
  .stButton > button { background-color: #01696f !important; color: white !important; border: none !important; border-radius: 4px !important; font-weight: 600 !important; padding: 8px 20px !important; }
  .stButton > button:hover { background-color: #0c4e54 !important; }

  /* Input fields */
  .stTextInput > div > div > input { background-color: #3c3c3c !important; color: #d4d4d4 !important; border: 1px solid #3e3e3e !important; border-radius: 4px !important; }
  .stTextInput > div > div > input:focus { border-color: #01696f !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #252526; border-bottom: 1px solid #3e3e3e; gap: 0; }
  .stTabs [data-baseweb="tab"] { background-color: transparent; color: #858585; border-radius: 0; padding: 10px 20px; font-size: 0.85rem; font-family: 'Courier New', monospace; }
  .stTabs [aria-selected="true"] { background-color: #1e1e1e !important; color: #4ec9b0 !important; border-bottom: 2px solid #01696f !important; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #3e3e3e; border-radius: 6px; }

  /* Sidebar text */
  .stSidebar * { color: #d4d4d4 !important; }

  /* Section headers */
  h1, h2, h3 { color: #d4d4d4 !important; letter-spacing: 0.05em; }

  /* Progress bars */
  .stProgress > div > div { background-color: #01696f !important; }

  /* Expander */
  .streamlit-expanderHeader { background-color: #2d2d2d !important; border: 1px solid #3e3e3e !important; color: #d4d4d4 !important; }

  /* Code blocks */
  .stCode { background-color: #1e1e1e !important; border: 1px solid #3e3e3e !important; }

  /* Bottom status bar effect */
  .stApp > footer { display: none; }
  .status-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #007acc; color: white; padding: 2px 12px; font-size: 0.75rem; font-family: 'Courier New', monospace; z-index: 999; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "repo_path": None,
    "repo_name": None,
    "file_list": None,
    "chunks": None,
    "graph": None,
    "collection": None,
    "stats": None,
    "query_log": [],
    "query_input": "",
    "followups": [],
    "last_results": [],
    "last_answer": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_logger = QueryLogger()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 CodeLens")
    st.caption("RAG-powered code intelligence")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("📁 ingestion.dash")
    st.markdown("🔍 query.console")
    st.markdown("📊 metrics.live")
    st.markdown("---")
    if st.session_state.repo_name:
        st.success(f"✓ {st.session_state.repo_name}")
        if st.session_state.stats:
            st.caption(
                f"{st.session_state.stats.total_files} files · "
                f"{st.session_state.stats.total_lines:,} lines"
            )
    else:
        st.warning("No repo loaded")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📁 ingestion.dash", "🔍 query.console", "📊 metrics.live"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ingestion
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Repository Analysis")

    url_input = st.text_input(
        "GitHub URL or local path",
        placeholder="https://github.com/owner/repo",
        key="url_input",
    )

    if st.button("Analyze Repo", type="primary", disabled=not url_input):
        try:
            with st.status("Running pipeline...", expanded=True) as status:

                status.update(label="Clone repository", state="running")
                local_path, repo_name = accept_input(url_input)
                st.write(f"Resolved: `{local_path}`")

                status.update(label="Detect languages", state="running")
                file_list, stats = walk_files(local_path)
                lang_summary = ", ".join(
                    f"{lang} ({n})" for lang, n in stats.languages.items()
                )
                st.write(f"Found {stats.total_files} files — {lang_summary}")

                status.update(label="Embed chunks", state="running")
                chunks = chunk_repo(local_path, file_list)
                collection = init_chromadb()
                embedded = embed_chunks(chunks)
                store_chunks(collection, embedded)
                st.write(f"Embedded {len(chunks)} chunks into ChromaDB")

                status.update(label="Build graph", state="running")
                graph = build_graph(chunks)
                st.write(
                    f"Graph built — {graph.number_of_nodes()} nodes, "
                    f"{graph.number_of_edges()} edges"
                )

                status.update(label="Complete!", state="complete")

            st.session_state.repo_path = local_path
            st.session_state.repo_name = repo_name
            st.session_state.file_list = file_list
            st.session_state.chunks = chunks
            st.session_state.graph = graph
            st.session_state.collection = collection
            st.session_state.stats = stats

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    # ── Metrics cards ──────────────────────────────────────────────────────────
    if st.session_state.chunks:
        _chunks = st.session_state.chunks
        _stats = st.session_state.stats
        _unique = len({c.source for c in _chunks})
        _coverage = round(_unique / max(_stats.total_files, 1) * 100, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Files", _stats.total_files)
        c2.metric("Embedded Chunks", len(_chunks))
        c3.metric("Lines of Code", f"{_stats.total_lines:,}")
        c4.metric("Coverage", f"{_coverage}%")

        # ── Language breakdown ─────────────────────────────────────────────────
        st.markdown("#### Language Breakdown")
        _total_files = sum(_stats.languages.values()) or 1
        for lang, count in sorted(_stats.languages.items(), key=lambda x: -x[1]):
            pct = count / _total_files
            st.progress(pct, text=f"{lang}  {count} files ({pct * 100:.1f}%)")

        # ── File table ─────────────────────────────────────────────────────────
        st.markdown("#### Indexed Files")
        _root = Path(st.session_state.repo_path)
        _file_data: dict[str, dict] = {}
        for _fp, _lang in (st.session_state.file_list or []):
            _rel = str(Path(_fp).relative_to(_root))
            _file_data[_rel] = {"LANG": _lang, "LOC": 0, "CHUNKS": 0}
        for chunk in _chunks:
            fd = _file_data.setdefault(chunk.source, {"LANG": chunk.language, "LOC": 0, "CHUNKS": 0})
            fd["LANG"] = chunk.language
            fd["CHUNKS"] += 1
            fd["LOC"] = max(fd["LOC"], chunk.start_line + chunk.text.count("\n"))

        st.dataframe(
            pd.DataFrame([
                {"PATH": src, **d}
                for src, d in sorted(_file_data.items())
            ]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "PATH": st.column_config.TextColumn("Path", width="large"),
                "LANG": st.column_config.TextColumn("Language"),
                "LOC": st.column_config.NumberColumn("Lines"),
                "CHUNKS": st.column_config.NumberColumn("Chunks"),
            },
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Query Console
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── Quick stats ────────────────────────────────────────────────────────────
    _qs_stats = _logger.get_stats()
    _g = st.session_state.graph
    _s = st.session_state.stats
    qs = st.columns(5)
    qs[0].metric("Files Indexed", _s.total_files if _s else 0)
    qs[1].metric("Chunks Embedded", len(st.session_state.chunks) if st.session_state.chunks else 0)
    qs[2].metric("Graph Nodes", _g.number_of_nodes() if _g else 0)
    qs[3].metric("Graph Edges", _g.number_of_edges() if _g else 0)
    qs[4].metric("Queries Logged", _qs_stats["total_queries"])

    st.markdown("---")

    # ── Query input ────────────────────────────────────────────────────────────
    query = st.text_input(
        "Ask about the codebase",
        placeholder="What does the authentication module do?",
        key="query_input",
    )

    _available_langs = (
        list(st.session_state.stats.languages.keys())
        if st.session_state.stats
        else ["python", "javascript", "typescript", "java", "cpp", "go", "tsx"]
    )
    lang_filter = st.multiselect(
        "Scope: filter by language",
        options=_available_langs,
        default=[],
        key="lang_filter",
    )

    _can_query = bool(query and st.session_state.collection)
    submit = st.button("Submit", type="primary", disabled=not _can_query)

    if not st.session_state.collection:
        st.info("Analyze a repository in the **Ingestion** tab first.")

    if submit and _can_query:
        try:
            _model = load_model()
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

            with st.spinner("Generating follow-up questions..."):
                _followups = generate_followups(query, _resp["answer"])

            _latency_ms = (time.perf_counter() - t0) * 1000
            _relevance = answer_relevance_score(query, _resp["answer"])

            st.session_state.last_results = _results
            st.session_state.last_answer = _resp["answer"]
            st.session_state.followups = _followups

            _entry = log_query(query, _resp["answer"], _results, _latency_ms, _relevance)
            _entry["repository"] = st.session_state.repo_name or ""
            _logger.log(_entry)

        except Exception as exc:
            st.error(f"Query failed: {exc}")

    # ── Answer ─────────────────────────────────────────────────────────────────
    if st.session_state.last_answer:
        st.markdown("#### Answer")
        st.markdown(st.session_state.last_answer)

        # ── Sources ───────────────────────────────────────────────────────────
        if st.session_state.last_results:
            st.markdown("#### Sources")
            for _r in st.session_state.last_results:
                _label = (
                    f"`{_r['source']}` · {_r['language']} · "
                    f"score {_r['score']:.2f}"
                )
                with st.expander(_label):
                    st.progress(
                        min(1.0, float(_r["score"])),
                        text=f"Relevance: {_r['score']:.2f}",
                    )
                    st.code(_r["text"], language=_r.get("language", "text"))

        # ── Follow-ups ────────────────────────────────────────────────────────
        if st.session_state.followups:
            st.markdown("#### Suggested Follow-ups")
            _fu_cols = st.columns(len(st.session_state.followups))
            for _i, (_col, _fq) in enumerate(
                zip(_fu_cols, st.session_state.followups)
            ):
                with _col:
                    if st.button(_fq, key=f"fu_{_i}", use_container_width=True):
                        st.session_state.query_input = _fq
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    _entries = _logger.load()
    _all_stats = _logger.get_stats()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision@5", f"{_all_stats['pass_rate']:.1%}")
    k2.metric("Avg Relevance", f"{_all_stats['avg_relevance']:.2f}")
    k3.metric("Median Latency", f"{_all_stats['avg_latency_ms']:.0f} ms")
    k4.metric("Total Queries", _all_stats["total_queries"])

    if not _entries:
        st.info("No queries logged yet. Use the **Query Console** to get started.")
    else:
        # ── Heatmap ───────────────────────────────────────────────────────────
        st.markdown("#### Query Volume")
        _day_names = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday",
        ]
        _heat: dict[tuple[int, int], int] = defaultdict(int)
        for _e in _entries:
            try:
                _dt = datetime.fromisoformat(_e["timestamp"])
                _heat[(_dt.weekday(), _dt.hour)] += 1
            except (ValueError, KeyError):
                pass

        _z = [
            [_heat.get((_day, _hr), 0) for _hr in range(24)]
            for _day in range(7)
        ]
        _fig = go.Figure(
            go.Heatmap(
                z=_z,
                x=list(range(24)),
                y=_day_names,
                colorscale="Blues",
                showscale=True,
            )
        )
        _fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(_fig, use_container_width=True)

        # ── Query history table ───────────────────────────────────────────────
        st.markdown("#### Query History")
        _BADGE = {
            "passed": "✅ passed",
            "flagged": "⚠️ flagged",
            "failed": "❌ failed",
        }
        _df = pd.DataFrame(_entries)
        _df["check_state"] = _df["check_state"].map(_BADGE).fillna(_df["check_state"])
        _df["timestamp"] = (
            pd.to_datetime(_df["timestamp"], utc=True)
            .dt.tz_convert(None)
            .dt.strftime("%Y-%m-%d %H:%M")
        )
        _show_cols = [
            c for c in
            ["timestamp", "query", "repository", "check_state",
             "relevance_score", "latency_ms", "sources_count"]
            if c in _df.columns
        ]
        st.dataframe(
            _df[_show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Time"),
                "query": st.column_config.TextColumn("Query", width="large"),
                "repository": st.column_config.TextColumn("Repo"),
                "check_state": st.column_config.TextColumn("Status"),
                "relevance_score": st.column_config.NumberColumn(
                    "Relevance", format="%.2f"
                ),
                "latency_ms": st.column_config.NumberColumn(
                    "Latency (ms)", format="%.0f"
                ),
                "sources_count": st.column_config.NumberColumn("Sources"),
            },
        )

st.markdown('<div class="status-bar">● main &nbsp;&nbsp; CodeLens &nbsp;&nbsp; RAG-powered code intelligence &nbsp;&nbsp; ready</div>', unsafe_allow_html=True)
