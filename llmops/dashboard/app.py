"""Streamlit LLMOps monitoring dashboard for LoyaltyLens.

Run:
    streamlit run llmops/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_EVAL_RESULTS_DIR = Path("llmops/eval_results")
_DRIFT_RESULTS_DIR = Path("llmops/drift_results")
_HISTORY_FILE = Path("llmops/prompt_registry/history.json")
_ACTIVE_FILE = Path("llmops/prompt_registry/active.json")
_DEPLOY_LOG = Path("data/deploy_log.jsonl")

st.set_page_config(page_title="LoyaltyLens LLMOps", layout="wide")
st.title("LoyaltyLens LLMOps Dashboard")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_eval_history() -> pd.DataFrame:
    files = sorted(_EVAL_RESULTS_DIR.glob("eval_*.json"), reverse=True)
    rows = []
    for f in files[:20]:
        data = json.loads(f.read_text())
        rows.append(
            {
                "timestamp": data.get("timestamp", f.stem),
                "mean_score": data.get("mean_aggregate_score", data.get("mean_score", 0.0)),
                "passed": data.get("passed", False),
                "n": data.get("n", 0),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "mean_score", "passed", "n"])


def _load_latest_eval() -> dict[str, object]:
    files = sorted(_EVAL_RESULTS_DIR.glob("eval_*.json"), reverse=True)
    if not files:
        return {"mean_aggregate_score": 0.0, "passed": False, "timestamp": "—"}
    return json.loads(files[0].read_text())


def _load_drift_report() -> dict[str, object]:
    report_file = _DRIFT_RESULTS_DIR / "drift_report.json"
    if not report_file.exists():
        return {"psi": 0.0, "status": "unknown", "current_date": "—"}
    return json.loads(report_file.read_text())


def _load_prompt_history() -> pd.DataFrame:
    if not _HISTORY_FILE.exists():
        return pd.DataFrame(columns=["version", "activated_at", "previous"])
    history = json.loads(_HISTORY_FILE.read_text())
    return pd.DataFrame(history)


def _load_last_deploy() -> str:
    if not _DEPLOY_LOG.exists():
        return "No deployments recorded"
    lines = _DEPLOY_LOG.read_text().strip().splitlines()
    if not lines:
        return "No deployments recorded"
    last = json.loads(lines[-1])
    return f"v{last.get('model_version', '?')} → {last.get('env', '?')} at {last.get('deployed_at', '?')}"


# ── top metrics row ────────────────────────────────────────────────────────────

latest_eval = _load_latest_eval()
drift_report = _load_drift_report()
active_version = json.loads(_ACTIVE_FILE.read_text())["version"] if _ACTIVE_FILE.exists() else "—"

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Active Prompt Version",
    active_version,
)

eval_score = latest_eval.get("mean_aggregate_score", latest_eval.get("mean_score", 0.0))
eval_passed = latest_eval.get("passed", False)
col2.metric(
    "Latest Eval Score",
    f"{eval_score:.3f}",
    delta="PASS" if eval_passed else "FAIL",
    delta_color="normal" if eval_passed else "inverse",
)

psi = drift_report.get("psi", 0.0)
drift_status = str(drift_report.get("status", "unknown"))
status_color = {"ok": "🟢", "warning": "🟡", "critical": "🔴"}.get(drift_status, "⚪")
col3.metric(
    f"Drift PSI  {status_color} {drift_status.upper()}",
    f"{psi:.4f}",
)

col4.metric("Last Deploy", _load_last_deploy())

st.divider()

# ── eval score trend ──────────────────────────────────────────────────────────

st.subheader("Eval Score Trend (last 20 runs)")
eval_history = _load_eval_history()
if not eval_history.empty:
    chart_df = eval_history.set_index("timestamp")[["mean_score"]].rename(
        columns={"mean_score": "Aggregate Score"}
    )
    st.line_chart(chart_df)
    st.caption("Threshold: 0.75 — builds below this line fail the CI eval gate.")
else:
    st.info("No eval results found. Run `python llmops/eval_harness/run_eval.py` first.")

st.divider()

# ── drift feature breakdown ────────────────────────────────────────────────────

st.subheader("Drift Monitor")
left, right = st.columns([1, 2])
left.markdown(
    f"**PSI:** {psi:.4f}  \n"
    f"**Status:** {status_color} **{drift_status.upper()}**  \n"
    f"**Checked:** {drift_report.get('current_date', '—')}  \n"
    f"**Baseline:** {drift_report.get('baseline_date', '—')}  \n"
    f"**N baseline / current:** {drift_report.get('n_baseline', '—')} / {drift_report.get('n_current', '—')}"
)
feature_breakdown = drift_report.get("feature_breakdown", {})
if feature_breakdown:
    breakdown_df = pd.DataFrame(
        list(feature_breakdown.items()), columns=["Feature", "PSI"]
    ).sort_values("PSI", ascending=False)
    right.dataframe(breakdown_df, use_container_width=True)
else:
    right.info("No per-feature breakdown available.")

st.divider()

# ── prompt version history ─────────────────────────────────────────────────────

st.subheader("Prompt Version History")
prompt_history = _load_prompt_history()
if not prompt_history.empty:
    st.dataframe(prompt_history, use_container_width=True)
else:
    st.info("No prompt history found.")
