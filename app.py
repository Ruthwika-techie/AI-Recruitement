"""
TechVest Recruitment Agent - Streamlit App (Multi-Agent Crew)
====================================================================
Tabs:
  1. Run Crew      — full 5-agent pipeline with live phase status
  2. Agent Trace   — per-agent step replay with agent colour coding
  3. Verifier      — peer-to-peer verification panel (borderline only)
  4. Audit Log     — metrics, injection flags, downloadable JSON
  5. Evaluation    — dataset / trace / output / red-team / governance

UI NOTE: all styling lives in THEME_CSS + the render_* helpers below.
Functional logic (crew invocation, session state, guardrails) is
unchanged from the original app — only presentation was reworked.
"""

import datetime
import json
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from data import CANDIDATES, JD, RUBRIC
from crew_graph import build_crew, make_config, make_inputs
from guardrails import fairness_ok, persist_audit

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TechVest Multi-Agent Crew",
    page_icon="🧭",
    layout="wide",
)

# ─────────────────────────────────────────────
# Theme (colors / pills / cards) — pure CSS, no logic
# Vibrant palette: plum + coral + amber, no navy/blue wash.
# ─────────────────────────────────────────────
PLUM        = "#2E1065"   # deep violet — sidebar / header base
PLUM_CARD   = "#3B1878"   # sidebar card fill
PLUM_BORDER = "#5B2A9E"   # sidebar card border
CORAL       = "#FF6B4A"   # primary accent / CTA
CORAL_DARK  = "#E5502F"   # CTA hover
PINK        = "#EC4899"   # secondary accent
AMBER_BG    = "#FFF3D6"
AMBER_BORDER= "#F4C542"
AMBER_TEXT  = "#8A5A00"
GREEN_BG    = "#E4F8EE"
GREEN_TEXT  = "#0E8A5F"
RED_BG      = "#FDEAEA"
RED_TEXT    = "#D93636"
PAGE_BG     = "#FFFAF5"   # warm off-white

THEME_CSS = f"""
<style>
/* ---- page & app shell ---- */
.stApp {{ background: {PAGE_BG}; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PLUM} 0%, #1D0B4A 100%);
    border-right: 1px solid {PLUM_BORDER};
}}
section[data-testid="stSidebar"] * {{ color: #F4EEFF !important; }}
section[data-testid="stSidebar"] hr {{ border-color: {PLUM_BORDER}; }}

/* ---- sidebar section "cards" ---- */
div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PLUM_CARD};
    border: 1px solid {PLUM_BORDER};
    border-radius: 12px;
    padding: 10px 12px 12px 12px;
    margin-bottom: 10px;
    box-sizing: border-box;
    overflow: hidden;
}}
/* Remove Streamlit's inner element margin inside sidebar cards */
div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlockBorderWrapper"]
  > div > div[data-testid="stVerticalBlock"] > div {{
    gap: 0 !important;
}}
div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlockBorderWrapper"]
  .stMarkdown {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}
.side-card-title {{
    font-size: 0.68em;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #C9B6F2;
    margin: 0 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {PLUM_BORDER};
    text-transform: uppercase;
    display: block;
}}
.side-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.86em;
    padding: 3px 0;
    color: #F4EEFF;
}}
.side-row .k {{ color: #C9B6F2; }}
.pill {{
    display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:0.72em; font-weight:700;
}}
.pill-coral {{ background:{CORAL}26; color:{CORAL}; border:1px solid {CORAL}66; }}
.pill-green {{ background:{GREEN_BG}; color:{GREEN_TEXT}; }}
.pill-amber {{ background:{AMBER_BG}; color:{AMBER_TEXT}; }}
.pill-red   {{ background:{RED_BG}; color:{RED_TEXT}; }}

/* ---- top header bar ---- */
.app-header {{
    background: linear-gradient(120deg, {PLUM} 0%, {PINK} 100%);
    border-radius: 16px;
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(46, 16, 101, 0.18);
}}
.app-header h1 {{ font-size: 1.4em; margin: 0 0 4px 0; color: white; }}
.app-header p {{ margin: 0; color: #F1E4FF; font-size: 0.88em; }}
.run-badge {{
    text-align:right; font-size:0.85em; color:#F1E4FF;
}}
.run-badge b {{ color: white; }}
.run-badge .status-ok {{ color:#FFD37A; font-weight:800; }}

/* ---- shortlist summary banner ---- */
.summary-banner {{
    background: linear-gradient(90deg, {GREEN_BG} 0%, #EAFBF2 100%);
    border: 1px solid #A9E7C6;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 0.92em;
    color: #0E5A3D;
}}

/* ---- candidate cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"].candidate-card-wrap {{
    border-radius: 14px !important;
}}
.rank-circle {{
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, {CORAL} 0%, {PINK} 100%);
    color: white; font-weight: 800;
    display:flex; align-items:center; justify-content:center;
    font-size: 0.9em; flex-shrink:0;
}}
.cand-name {{ font-weight: 800; font-size: 1.05em; color: {PLUM}; }}
.cand-score {{ color: #8A7A9E; font-size: 0.88em; margin-left: 4px; }}
.cand-desc {{ font-size: 0.9em; color: #4A3F5C; margin-top: 4px; line-height:1.5; }}
.chip {{
    display:inline-block; background:#FBEEE6; color:{CORAL_DARK};
    border-radius: 8px; padding: 2px 9px; font-size: 0.78em;
    margin-right: 6px; margin-top: 8px; font-weight:700;
}}

/* ---- verdict pills ---- */
.verdict-pill {{
    padding: 4px 12px; border-radius: 999px; font-weight: 800;
    font-size: 0.74em; white-space: nowrap;
}}

/* ---- approval gate card ---- */
.gate-card {{
    background: {AMBER_BG};
    border: 1px solid {AMBER_BORDER};
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 10px;
}}
.gate-card b {{ color: #6B4200; }}

/* buttons */
div.stButton > button[kind="primary"] {{
    background: linear-gradient(120deg, {CORAL} 0%, {PINK} 100%);
    border: none;
    color: white;
    font-weight: 700;
}}
div.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(120deg, {CORAL_DARK} 0%, {PINK} 100%);
}}
div.stButton > button:not([kind="primary"]) {{
    border: 1px solid {PLUM_BORDER};
    color: {PLUM};
    font-weight: 600;
}}

/* ---- tabs: bigger, clearer, on-brand ---- */
button[data-baseweb="tab"] {{
    font-weight: 700;
    font-size: 0.95em;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {CORAL_DARK} !important;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {CORAL} !important;
}}

/* ---- metrics ---- */
div[data-testid="stMetricValue"] {{ color: {PLUM}; font-weight: 800; }}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "run_id":        None,
        "crew_graph":    None,
        "cfg":           None,
        "phase":         "idle",   # idle | running | gate | done | escalated
        "state_at_gate": None,
        "final_state":   None,
        "audit_path":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# Style constants used by helpers
# ─────────────────────────────────────────────
VERDICT_STYLE = {
    "interview": {"pill_bg": GREEN_BG, "pill_text": GREEN_TEXT, "label": "✅ INTERVIEW"},
    "hold":      {"pill_bg": AMBER_BG, "pill_text": AMBER_TEXT, "label": "⏸️ HOLD"},
    "reject":    {"pill_bg": RED_BG,   "pill_text": RED_TEXT,   "label": "❌ NOT A FIT — THIS ROLE"},
}

AGENT_COLOR = {
    "coordinator": "#8B5CF6",   # violet
    "analyst":     "#FB923C",   # amber-orange
    "scorer":      "#14B8A6",   # teal
    "verifier":    "#EC4899",   # pink
    "decider":     "#D97706",   # deep amber
    "scheduler":   "#22C55E",   # green
}

RUBRIC_LABELS = {
    "python_ml": "Python / ML core",
    "projects":  "Relevant projects",
    "tooling":   "Hands-on tooling",
    "communication": "Communication",
}


def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:0.85em">{text}</span>'
    )


def verdict_pill(verdict: str) -> str:
    st_ = VERDICT_STYLE.get(verdict, {"pill_bg": "#eee", "pill_text": "#555", "label": verdict.upper()})
    return (
        f'<span class="verdict-pill" style="background:{st_["pill_bg"]};'
        f'color:{st_["pill_text"]}">{st_["label"]}</span>'
    )


def agent_badge(agent: str) -> str:
    return badge(agent.upper(), AGENT_COLOR.get(agent, "#555"))


def get_active_state() -> dict | None:
    if st.session_state.final_state is not None:
        return st.session_state.final_state
    if st.session_state.state_at_gate is not None:
        return st.session_state.state_at_gate
    return None


def reset_session():
    for k, v in {
        "run_id": None, "crew_graph": None, "cfg": None,
        "phase": "idle", "state_at_gate": None,
        "final_state": None, "audit_path": None,
    }.items():
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Render helpers — candidate cards (mockup style)
# ─────────────────────────────────────────────
def render_scorecard(card: dict):
    """Detailed table view kept inside the 'view trajectory' expander."""
    rows = ""
    for cs in card.get("scores", []):
        filled = min(cs["score"], 5)
        bar    = "█" * filled + "░" * (5 - filled)
        ev     = cs["evidence"][:90].replace("<", "&lt;").replace(">", "&gt;")
        rows  += (
            f"<tr>"
            f"<td style='padding:4px 10px'>{cs['criterion']}</td>"
            f"<td style='padding:4px 10px;font-family:monospace;color:{CORAL_DARK}'>{bar} {cs['score']}/5</td>"
            f"<td style='padding:4px 10px;color:#555;font-size:0.85em'>{ev}</td>"
            f"</tr>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
        f"<thead><tr style='background:{AMBER_BG}'>"
        "<th style='padding:4px 10px;text-align:left'>Criterion</th>"
        "<th style='padding:4px 10px;text-align:left'>Score</th>"
        "<th style='padding:4px 10px;text-align:left'>Evidence</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


def _score_chips(card: dict) -> str:
    chips = ""
    for cs in card.get("scores", []):
        label = cs["criterion"].split()[0][:10]
        chips += f'<span class="chip">{label} <b>{cs["score"]}</b></span>'
    return chips


def render_shortlist(shortlist: list, show_slot: bool = False):
    if not shortlist:
        st.info("No decisions yet.")
        return

    for i, d in enumerate(shortlist, 1):
        verdict  = d.get("verdict", "unknown")
        name     = d.get("name", "Unknown")
        score    = d.get("weighted", 0.0)
        verified = d.get("verified", True)

        with st.container(border=True):
            top = st.columns([0.06, 0.7, 0.24])
            top[0].markdown(f'<div class="rank-circle">{i}</div>', unsafe_allow_html=True)
            top[1].markdown(
                f'<span class="cand-name">{name}</span>'
                f'<span class="cand-score"> · weighted score {score:.1f} / 5</span>',
                unsafe_allow_html=True,
            )
            top[2].markdown(
                f'<div style="text-align:right">{verdict_pill(verdict)}</div>',
                unsafe_allow_html=True,
            )

            if not verified:
                st.markdown(
                    '<span class="pill pill-amber">⚠️ verification flagged — review recommended</span>',
                    unsafe_allow_html=True,
                )

            justification = d.get("justification", "N/A")
            st.markdown(f'<div class="cand-desc">{justification}</div>', unsafe_allow_html=True)
            st.markdown(_score_chips(d["scorecard"]), unsafe_allow_html=True)

            if show_slot and d.get("proposed_slot"):
                st.success(f"📅 Interview slot booked: **{d['proposed_slot']}**")

            with st.expander("🔎 View full scorecard & trajectory"):
                vn = d.get("verifier_note", "")
                if vn:
                    st.caption(f"⚖️ Verifier: {vn}")
                render_scorecard(d["scorecard"])


def render_agent_trace(trace: list, max_steps: int | None = None):
    entries = trace[:max_steps] if max_steps else trace
    for i, entry in enumerate(entries, 1):
        agent = entry.get("agent", "unknown")
        color = AGENT_COLOR.get(agent, "#555")
        cols  = st.columns([0.08, 0.92])
        cols[0].markdown(
            f'<div style="text-align:center">'
            f'<b style="font-size:0.8em">{i:02d}</b><br>'
            f'<span style="background:{color};color:white;padding:1px 5px;'
            f'border-radius:6px;font-size:0.7em">{agent}</span></div>',
            unsafe_allow_html=True,
        )
        parts = []
        for k in ("thought", "action", "observation"):
            v = entry.get(k, "")
            if v:
                val = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
                val = val[:160] + "..." if len(val) > 160 else val
                parts.append(f"**{k}:** {val}")
        cols[1].markdown("  \n".join(parts))
        st.divider()


def _last_run_metrics(state: dict | None):
    """Best-effort derivation of step/tool/latency stats for the sidebar.
    Some of these (latency) aren't tracked in the underlying state today —
    shown as '—' until that's wired up."""
    if not state:
        return {"steps": "—", "tools": "—", "latency": "—"}
    trace = state.get("agent_trace", [])
    steps = len(trace)
    tools = sum(1 for e in trace if e.get("action"))
    latency = state.get("latency_seconds")
    latency_str = f"{latency:.1f}s" if isinstance(latency, (int, float)) else "—"
    return {"steps": steps, "tools": tools, "latency": latency_str}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Brand header ──────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;
            padding:12px 0 14px 0;
            border-bottom:1px solid #5B2A9E;
            margin-bottom:12px;">
  <div style="background:linear-gradient(135deg,#FF6B4A,#EC4899);
              border-radius:10px;min-width:36px;width:36px;height:36px;
              display:flex;align-items:center;justify-content:center;
              font-size:1.1em;flex-shrink:0;">🧭</div>
  <div style="min-width:0;">
    <div style="font-weight:800;font-size:0.98em;color:#FFD37A;
                line-height:1.3;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;">TechVest Crew</div>
    <div style="font-size:0.70em;color:#C9B6F2;white-space:nowrap;">
        Multi-Agent Recruitment</div>
  </div>
</div>
""", unsafe_allow_html=True)

    active_state = get_active_state()

    # ── shared inline styles ──────────────────
    _ROW  = ("display:flex;justify-content:space-between;align-items:center;"
             "padding:5px 0;border-bottom:1px solid rgba(91,42,158,0.3);")
    _ROWL = ("display:flex;justify-content:space-between;align-items:center;"
             "padding:5px 0;")                     # last row — no border
    _LBL  = "font-size:0.81em;color:#C9B6F2;white-space:nowrap;flex-shrink:0;"
    _VAL  = "font-size:0.81em;font-weight:700;color:#F4EEFF;"

    # ── Run Config ────────────────────────────
    with st.container(border=True):
        st.markdown(f"""
<div class="side-card-title">⚙️ Run Config</div>
<div style="{_ROW}">
  <span style="{_LBL}">Role</span>
  <span style="font-size:0.80em;font-weight:700;color:#FFD37A;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
               margin-left:8px;text-align:right;">Junior AI Engineer</span>
</div>
<div style="{_ROW}">
  <span style="{_LBL}">Candidates</span>
  <span style="background:#5B2A9E;color:white;font-weight:700;
               border-radius:999px;padding:1px 9px;font-size:0.74em;
               white-space:nowrap;">{len(CANDIDATES)}</span>
</div>
<div style="{_ROWL}">
  <span style="{_LBL}">Framework</span>
  <span class="pill pill-coral" style="margin-left:8px;">LangGraph</span>
</div>
""", unsafe_allow_html=True)

    # ── Scoring Rubric ────────────────────────
    with st.container(border=True):
        rubric_colors = ["#FF6B4A", "#EC4899", "#A78BFA", "#38BDF8"]
        rubric_items  = list(RUBRIC.items())
        rows_html = ""
        for idx, (c, w) in enumerate(rubric_items):
            label   = RUBRIC_LABELS.get(c, c.replace("_", " ").title())
            color   = rubric_colors[idx % len(rubric_colors)]
            pct     = int(w * 100)
            is_last = idx == len(rubric_items) - 1
            mb      = "0px" if is_last else "8px"
            rows_html += f"""
<div style="margin-bottom:{mb};">
  <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:3px;">
    <span style="font-size:0.80em;color:#E8DEFF;white-space:nowrap;
                 overflow:hidden;text-overflow:ellipsis;max-width:72%;">{label}</span>
    <span style="font-size:0.76em;font-weight:700;color:{color};
                 white-space:nowrap;margin-left:6px;">×{w}</span>
  </div>
  <div style="width:100%;background:rgba(91,42,158,0.35);
              border-radius:999px;height:4px;overflow:hidden;
              box-sizing:border-box;">
    <div style="background:{color};width:{pct}%;height:100%;
                border-radius:999px;"></div>
  </div>
</div>"""
        st.markdown(f"""
<div class="side-card-title">📐 Scoring Rubric</div>
{rows_html}
""", unsafe_allow_html=True)

    # ── Guardrails ────────────────────────────
    with st.container(border=True):
        flags         = active_state.get("injection_flags", []) if active_state else []
        step_cap_used = len(active_state.get("agent_trace", [])) if active_state else 0
        step_pct      = min(100, int(step_cap_used / 25 * 100))
        step_color    = ("#FF6B4A" if step_pct > 75
                         else "#38BDF8" if step_pct > 40
                         else "#0E8A5F")

        fair_ok     = active_state is not None and not flags
        fair_label  = "PASS" if fair_ok else "—"
        fair_color  = "#0E8A5F" if fair_ok else "#8A5A00"
        fair_bg     = "rgba(14,138,95,0.15)" if fair_ok else "rgba(244,197,66,0.15)"
        fair_border = "rgba(14,138,95,0.4)"  if fair_ok else "rgba(244,197,66,0.4)"

        _DOT = ("<span style='display:inline-flex;align-items:center;"
                "justify-content:center;width:8px;height:8px;border-radius:50%;"
                "background:#0E8A5F;box-shadow:0 0 5px rgba(14,138,95,0.7);"
                "flex-shrink:0;'></span>")
        _ACT = ("<span style='font-size:0.74em;font-weight:700;"
                "color:#0E8A5F;white-space:nowrap;'>ACTIVE</span>")

        st.markdown(f"""
<div class="side-card-title">🛡️ Guardrails</div>
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:4px;">
    <span style="{_LBL}">Step budget</span>
    <span style="font-size:0.76em;font-weight:700;
                 color:{step_color};white-space:nowrap;">{step_cap_used} / 25</span>
  </div>
  <div style="width:100%;background:rgba(91,42,158,0.35);
              border-radius:999px;height:5px;overflow:hidden;box-sizing:border-box;">
    <div style="background:{step_color};width:{step_pct}%;
                height:100%;border-radius:999px;"></div>
  </div>
</div>
<div style="{_ROW}">
  <span style="{_LBL}">Human gate</span>
  <span style="display:inline-flex;align-items:center;gap:5px;">{_DOT}{_ACT}</span>
</div>
<div style="{_ROW}">
  <span style="{_LBL}">Injection defence</span>
  <span style="display:inline-flex;align-items:center;gap:5px;">{_DOT}{_ACT}</span>
</div>
<div style="{_ROWL}">
  <span style="{_LBL}">Fairness check</span>
  <span style="font-size:0.74em;font-weight:700;color:{fair_color};
               background:{fair_bg};border:1px solid {fair_border};
               padding:1px 9px;border-radius:999px;white-space:nowrap;">
    {fair_label}</span>
</div>
""", unsafe_allow_html=True)

    # ── Last Run Metrics ──────────────────────
    with st.container(border=True):
        m = _last_run_metrics(active_state)
        items = [
            ("🔢", "Steps taken", m["steps"]),
            ("🔧", "Tool calls",  m["tools"]),
            ("⚡", "Latency",     m["latency"]),
        ]
        rows_html = ""
        for i, (icon, label, value) in enumerate(items):
            style = _ROWL if i == len(items) - 1 else _ROW
            rows_html += f"""
<div style="{style}">
  <span style="{_LBL}">{icon}&nbsp;{label}</span>
  <span style="font-size:0.81em;font-weight:700;color:#FFD37A;
               white-space:nowrap;">{value}</span>
</div>"""
        st.markdown(f"""
<div class="side-card-title">📊 Last Run</div>
{rows_html}
""", unsafe_allow_html=True)

    # ── Candidates & options (outside cards) ─
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    with st.expander("👥 Candidates in this run"):
        for i, c in enumerate(CANDIDATES, 1):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
                f'<span style="background:#5B2A9E;color:white;border-radius:50%;'
                f'min-width:20px;width:20px;height:20px;display:inline-flex;'
                f'align-items:center;justify-content:center;'
                f'font-size:0.69em;font-weight:700;flex-shrink:0;">{i}</span>'
                f'<span style="font-size:0.83em;">{c["name"]}</span></div>',
                unsafe_allow_html=True,
            )

    run_fairness = st.checkbox("Run fairness check (name-swap)", value=False,
                               help="~2 extra LLM calls per candidate")

    with st.expander("🗺️ Agent graph"):
        st.markdown("""
```
coordinator (supervisor)
     |
  analyst ──▶ scorer
                |
  (borderline?) ──▶ verifier
                |         |
             decider ◀────┘
                |
    [human gate] ──▶ scheduler
```
""")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
n_candidates = len(CANDIDATES)
phase = st.session_state.phase
status_label = {
    "idle": "not started",
    "running": "running…",
    "gate": "awaiting approval",
    "done": "completed ✓",
    "escalated": "escalated",
}.get(phase, phase)

st.markdown(
    f'<div class="app-header">'
    f'<div><h1>🧭 Screening {n_candidates} candidates · Junior AI Engineer</h1>'
    f'<p>Agent chose its own tool order · trajectory logged · 1 action pending approval</p></div>'
    f'<div class="run-badge">Run #{st.session_state.run_id or "—"}<br>'
    f'<span class="status-ok">{status_label}</span></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_run, tab_trace, tab_verifier, tab_audit, tab_eval = st.tabs([
    "▶ Run Crew",
    "🔍 Agent Trace",
    "⚖️ Verifier",
    "📋 Audit Log",
    "🧪 Evaluation",
])


# ══════════════════════════════════════════════
# TAB 1 — Run Crew
# ══════════════════════════════════════════════
with tab_run:

    # IDLE ──────────────────────────────────────
    if st.session_state.phase == "idle":
        st.info(
            "Press **🚀 Start Crew** to run all 5 agents: "
            "Coordinator → Analyst → Scorer → Verifier (borderline) → Decider → Scheduler."
        )

        with st.expander("ℹ️ How this crew works", expanded=False):
            st.markdown("""
**Supervisor–Worker:** Coordinator sequences and caps the crew.
**Pipeline:** Analyst → Scorer (staged transform).
**Peer-to-peer:** Scorer ⇄ Verifier, only on borderline candidates (score 2.8–3.4).
**Feedback loop:** Verifier can send work back to Analyst/Scorer (max 3 revisions).
**Human gate:** Pauses before Scheduler; propose_interview never fires without approval.
**Step budget:** recursion_limit=25 across the whole crew.
""")

        if st.button("🚀 Start Crew", type="primary", use_container_width=True):
            st.session_state.run_id    = str(uuid.uuid4())[:8]
            st.session_state.crew_graph = build_crew(interrupt_before_scheduler=True)
            st.session_state.cfg = make_config(st.session_state.run_id)
            st.session_state.phase = "running"
            st.rerun()

    # RUNNING ───────────────────────────────────
    if st.session_state.phase == "running":
        inputs = make_inputs(JD, RUBRIC, CANDIDATES)

        with st.spinner("Running crew: coordinator, analyst, scorer, verifier, decider..."):
            try:
                st.session_state.crew_graph.invoke(inputs, config=st.session_state.cfg)
                snap = st.session_state.crew_graph.get_state(st.session_state.cfg)
                st.session_state.state_at_gate = snap.values

                if snap.next == ("scheduler",):
                    st.session_state.phase = "gate"
                elif not snap.next:
                    st.session_state.final_state = snap.values
                    st.session_state.phase = "done"
                else:
                    st.session_state.state_at_gate = snap.values
                    st.session_state.phase = "gate"

            except Exception as e:
                st.error(f"Crew error: {e}")
                st.session_state.phase = "idle"
        st.rerun()

    # GATE + DONE ───────────────────────────────
    if st.session_state.phase in ("gate", "done", "escalated"):
        state = get_active_state()

        if state is None:
            st.error("State lost — please restart.")
            if st.button("Reset"):
                reset_session()
                st.rerun()
        else:
            flags = state.get("injection_flags", [])
            rc = state.get("revision_count", 0)
            shortlist = state.get("shortlist", [])
            n_interview = sum(1 for d in shortlist if d["verdict"] == "interview")
            n_hold      = sum(1 for d in shortlist if d["verdict"] == "hold")
            n_reject    = sum(1 for d in shortlist if d["verdict"] == "reject")

            summary_bits = []
            if n_interview: summary_bits.append(f"{n_interview} to interview")
            if n_hold:      summary_bits.append(f"{n_hold} on hold")
            if n_reject:    summary_bits.append(f"{n_reject} not a fit")
            summary_text = ", ".join(summary_bits) if summary_bits else "no candidates scored yet"

            st.markdown(
                f'<div class="summary-banner">🛡️ <b>Shortlist:</b> {summary_text} for this role. '
                f'Every ranking below cites résumé evidence — expand any candidate to see the full trajectory.</div>',
                unsafe_allow_html=True,
            )

            if flags:
                st.markdown(
                    f'<span class="pill pill-amber">⚠️ Guardrail 3 — prompt injection detected in: '
                    f'{", ".join(flags)} — text treated as data, ranking unaffected.</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

            if rc > 0:
                st.caption(f"🔁 Verifier triggered **{rc} revision(s)** during this run.")

            render_shortlist(shortlist, show_slot=(st.session_state.phase == "done"))

            # ── HUMAN GATE ──────────────────────────
            if st.session_state.phase == "gate":
                candidates_for_interview = [
                    d["name"] for d in shortlist if d["verdict"] == "interview"
                ]
                slot_hint = ""
                if shortlist:
                    first = next((d for d in shortlist if d["verdict"] == "interview"), None)
                    if first and first.get("proposed_slot"):
                        slot_hint = f' <b>{first["proposed_slot"]}</b> slot.'

                gate_html = (
                    '<div class="gate-card">🔒 <b>Human approval required.</b> '
                )
                if candidates_for_interview:
                    gate_html += (
                        f'Agent proposes: interview <b>{", ".join(candidates_for_interview)}</b>,'
                        f'{slot_hint} <code>propose_interview</code> will not fire until you confirm.'
                    )
                else:
                    gate_html += "No candidates were marked for interview this run."
                gate_html += "</div>"
                st.markdown(gate_html, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("❌ Reject", use_container_width=True):
                        st.session_state.final_state = state
                        st.session_state.audit_path  = persist_audit(
                            {
                                "shortlist":       state.get("shortlist", []),
                                "injection_flags": state.get("injection_flags", []),
                                "trajectory":      state.get("agent_trace", []),
                            },
                            st.session_state.run_id + "_rejected",
                        )
                        st.session_state.phase = "done"
                        st.rerun()
                with col2:
                    if st.button("✅ Approve & schedule", type="primary", use_container_width=True):
                        with st.spinner("Scheduling..."):
                            try:
                                st.session_state.crew_graph.invoke(None, config=st.session_state.cfg)
                                final_snap = st.session_state.crew_graph.get_state(st.session_state.cfg)
                                st.session_state.final_state = final_snap.values
                                st.session_state.audit_path  = persist_audit(
                                    {
                                        "shortlist":       final_snap.values.get("shortlist", []),
                                        "injection_flags": final_snap.values.get("injection_flags", []),
                                        "trajectory":      final_snap.values.get("agent_trace", []),
                                    },
                                    st.session_state.run_id,
                                )
                                st.session_state.phase = "done"
                            except Exception as e:
                                st.error(f"Scheduling error: {e}")
                        st.rerun()

            # ── DONE ────────────────────────────────
            if st.session_state.phase == "done":
                st.divider()
                st.success(f"🎉 Crew run complete  |  run_id: `{st.session_state.run_id}`")

                if run_fairness:
                    st.subheader("⚖️ Fairness Check — Guardrail #4")
                    with st.spinner("Name-swap fairness test..."):
                        for c in CANDIDATES:
                            try:
                                ok = fairness_ok(
                                    resume_text=c["resume"],
                                    rubric=RUBRIC,
                                    original_name=c["name"].split()[0],
                                )
                                if ok:
                                    st.success(f"✅ {c['name']}: fair (name-swap score matches)")
                                else:
                                    st.error(f"❌ {c['name']}: fairness issue detected")
                            except Exception as e:
                                st.warning(f"Fairness check failed for {c['name']}: {e}")

                st.divider()
                if st.button("🔄 Start a new run", use_container_width=True):
                    reset_session()
                    st.rerun()


# ══════════════════════════════════════════════
# TAB 2 — Agent Trace
# ══════════════════════════════════════════════
with tab_trace:
    active = get_active_state()

    if active is None:
        st.info("Run the crew first to see the agent trace here.")
    else:
        trace = active.get("agent_trace", [])

        st.subheader(f"🔍 Agent Trace — {len(trace)} steps")
        st.caption("Every thought, action, and observation from each agent — colour-coded by agent.")

        legend_html = "  ".join(
            f'<span style="background:{c};color:white;padding:2px 8px;border-radius:8px;font-size:0.8em">{a}</span>'
            for a, c in AGENT_COLOR.items()
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown("")

        agents_in_trace = sorted(set(e.get("agent", "unknown") for e in trace))
        selected = st.multiselect(
            "Filter by agent:",
            options=agents_in_trace,
            default=agents_in_trace,
        )
        filtered = [e for e in trace if e.get("agent") in selected]

        if filtered:
            step = st.slider(
                "Replay up to step:",
                min_value=1,
                max_value=len(filtered),
                value=len(filtered),
                step=1,
            )
            render_agent_trace(filtered, max_steps=step)
        else:
            st.info("No entries for the selected agents.")


# ══════════════════════════════════════════════
# TAB 3 — Verifier Panel
# ══════════════════════════════════════════════
with tab_verifier:
    active = get_active_state()

    if active is None:
        st.info("Run the crew first to see verifier activity here.")
    else:
        st.subheader("⚖️ Peer-to-Peer Verifier")
        st.caption(
            "The Verifier re-checks candidates in the borderline band (2.8–3.4).  \n"
            "It runs a name-swap fairness re-score and checks injection-flag containment."
        )

        verified_scores = active.get("verified_scores", [])
        revision_count  = active.get("revision_count", 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total verified",   len(verified_scores))
        c2.metric("Passed",           sum(1 for v in verified_scores if v.get("verified", True)))
        c3.metric("Flagged",          sum(1 for v in verified_scores if not v.get("verified", True)))
        c4.metric("Revisions",        revision_count)

        st.divider()

        if not verified_scores:
            st.info("No verifier activity recorded.")
        else:
            for v in verified_scores:
                name     = v["name"]
                ok       = v.get("verified", True)
                note     = v.get("verifier_note", "N/A")
                blind_w  = v.get("blind_weighted")
                weighted = v.get("weighted", 0.0)

                icon = "✅" if ok else "⚠️"
                with st.expander(
                    f"{icon}  {name}  |  Score: {weighted:.2f}  |  Verified: {ok}",
                    expanded=(not ok),
                ):
                    if "outside borderline band" in note:
                        st.success("Outside borderline band — passed through automatically.")
                    else:
                        col1, col2 = st.columns(2)
                        col1.metric("Original score", f"{weighted:.2f}")
                        if blind_w is not None:
                            col2.metric(
                                "Blind (name-swap) score",
                                f"{blind_w:.2f}",
                                delta=f"{blind_w - weighted:+.2f}",
                            )

                    st.caption(f"Verifier note: {note}")

                    if not ok:
                        st.error(
                            "Verification failed — score difference exceeded fairness threshold (0.5). "
                            "This candidate was sent back for re-scoring."
                        )

        if revision_count > 0:
            st.divider()
            st.markdown(f"**🔁 Revision log** — {revision_count} total revision(s)")
            verifier_trace = [
                e for e in active.get("agent_trace", [])
                if e.get("agent") == "verifier"
            ]
            for e in verifier_trace:
                obs = e.get("observation", "")
                thought = e.get("thought", "")
                icon = "✅" if "ok=True" in thought or "outside" in thought else "⚠️"
                st.markdown(f"{icon} `{thought[:90]}`")
                if obs:
                    st.caption(obs[:120])


# ══════════════════════════════════════════════
# TAB 4 — Audit Log
# ══════════════════════════════════════════════
with tab_audit:
    final = st.session_state.final_state

    if final is None:
        st.info("Run the crew first to see the audit log here.")
    else:
        payload = {
            "ts":              datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id":          st.session_state.run_id,
            "shortlist":       final.get("shortlist", []),
            "injection_flags": final.get("injection_flags", []),
            "revision_count":  final.get("revision_count", 0),
            "agent_trace":     final.get("agent_trace", []),
        }
        audit_json = json.dumps(payload, indent=2, default=str)

        st.subheader("📋 Audit Log — Guardrail #5")
        st.caption(
            "Full run saved to `audit/run_<id>.json`. "
            "Every decision can be reconstructed and explained from this file."
        )

        shortlist = final.get("shortlist", [])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Candidates",    len(shortlist))
        c2.metric("Interview",     sum(1 for d in shortlist if d["verdict"] == "interview"))
        c3.metric("Hold",          sum(1 for d in shortlist if d["verdict"] == "hold"))
        c4.metric("Reject",        sum(1 for d in shortlist if d["verdict"] == "reject"))
        c5.metric("Revisions",     final.get("revision_count", 0))

        st.divider()

        flags = final.get("injection_flags", [])
        if flags:
            st.error(f"Injection detected in: {', '.join(flags)}")
        else:
            st.success("No prompt injection detected.")

        if st.session_state.audit_path:
            st.markdown(f"**Saved to:** `{st.session_state.audit_path}`")

        with st.expander("View raw JSON", expanded=False):
            st.code(audit_json, language="json")

        st.download_button(
            label="⬇️ Download audit JSON",
            data=audit_json,
            file_name=f"run_{st.session_state.run_id}.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════
# TAB 5 — Evaluation Suite
# ══════════════════════════════════════════════
with tab_eval:
    st.subheader("🧪 Evaluation Suite")
    st.caption(
        "5 evaluation layers: dataset, trace invariants, output quality, "
        "red-team, and governance gate checks."
    )

    if "eval_results" not in st.session_state:
        st.session_state["eval_results"] = None
    if "eval_running" not in st.session_state:
        st.session_state["eval_running"] = False

    st.markdown("**Select layers to run:**")
    col_a, col_b, col_c, col_d = st.columns(4)
    run_l2 = col_a.checkbox("Layer 2: Trace",        value=True)
    run_l3 = col_b.checkbox("Layer 3: Output",       value=True)
    run_l4 = col_c.checkbox("Layer 4: Red-Team",     value=True)
    run_l5 = col_d.checkbox("Layer 5: Governance",   value=True)
    fast_mode = st.checkbox("Fast mode (skip LLM judge + fairness re-score)", value=False)

    with st.expander("📄 View 10-task evaluation dataset", expanded=False):
        try:
            with open("eval_dataset.json", encoding="utf-8") as f:
                dataset_json = json.load(f)
            for task in dataset_json:
                cat_color = {
                    "strong_fit": "#0E8A5F", "borderline": "#F4C542",
                    "weak_fit": "#D93636", "injection": "#D93636",
                    "missing_field": "#F4C542", "out_of_scope": "#8A7A9E",
                    "escalation": "#8B5CF6",
                }.get(task["category"], "#8A7A9E")
                st.markdown(
                    f"**[{task['id']}]** {task['description']} &nbsp;"
                    f'<span style="background:{cat_color};color:white;padding:1px 8px;'
                    f'border-radius:8px;font-size:0.8em">{task["category"]}</span>  '
                    f'— expected: **{task["expected_decision"]}**',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    st.divider()

    if st.button("🚀 Run Evaluation Suite", type="primary", use_container_width=True,
                 disabled=not (run_l2 or run_l3 or run_l4 or run_l5)):
        import json as _json

        with open("eval_dataset.json", encoding="utf-8") as _f:
            _dataset = _json.load(_f)

        from data import JD as _JD, RUBRIC as _RUBRIC
        from crew_graph import build_crew as _build_crew, make_config as _make_config, make_inputs as _make_inputs

        def _run_task(task):
            candidate = task["input"]
            _crew = _build_crew(interrupt_before_scheduler=True)
            _cfg  = _make_config(f"app_eval_{task['id']}_{uuid.uuid4().hex[:4]}")
            _inputs = _make_inputs(_JD, _RUBRIC, [candidate])
            _crew.invoke(_inputs, config=_cfg)
            _snap = _crew.get_state(_cfg)
            if _snap.next == ("scheduler",):
                _crew.invoke(None, config=_cfg)
            return _crew.get_state(_cfg).values

        layer_results = []

        if run_l2:
            from eval_trace import evaluate_trace
            l2_details = []
            prog = st.progress(0, text="Layer 2: Trace invariants...")
            for i, task in enumerate(_dataset):
                try:
                    state = _run_task(task)
                    r = evaluate_trace(
                        task,
                        state.get("agent_trace", []),
                        state.get("injection_flags", []),
                        state.get("shortlist", []),
                        use_judge=not fast_mode,
                    )
                    l2_details.append(r)
                except Exception:
                    pass
                prog.progress((i + 1) / len(_dataset), text=f"Layer 2: {i+1}/{len(_dataset)}")
            prog.empty()
            passed_l2 = sum(1 for r in l2_details if r.overall_pass)
            layer_results.append({
                "layer": 2, "name": "Trace & Tool-Call",
                "tasks_passed": passed_l2, "tasks_total": len(l2_details),
                "pass_rate": round(passed_l2 / len(l2_details), 2) if l2_details else 0,
                "invariant_pass_rate": round(sum(1 for r in l2_details if r.invariant_pass) / len(l2_details), 2) if l2_details else 0,
                "avg_tool_accuracy": round(sum(r.tool_call_accuracy for r in l2_details) / len(l2_details), 2) if l2_details else 0,
                "avg_judge_score": round(sum(r.judge_score for r in l2_details) / len(l2_details), 2) if l2_details else 0,
                "details": [{"task_id": r.task_id, "invariant_pass": r.invariant_pass,
                              "tool_accuracy": r.tool_call_accuracy, "judge_score": r.judge_score,
                              "overall": r.overall_pass, "failures": r.invariant_failures + r.tool_call_failures}
                             for r in l2_details],
            })

        if run_l3:
            from eval_output import evaluate_output
            l3_details = []
            prog = st.progress(0, text="Layer 3: Output quality...")
            for i, task in enumerate(_dataset):
                try:
                    state = _run_task(task)
                    r = evaluate_output(task, state.get("shortlist", []), run_fairness=not fast_mode)
                    l3_details.append(r)
                except Exception:
                    pass
                prog.progress((i + 1) / len(_dataset), text=f"Layer 3: {i+1}/{len(_dataset)}")
            prog.empty()
            passed_l3 = sum(1 for r in l3_details if r.overall_pass)
            layer_results.append({
                "layer": 3, "name": "Output Quality",
                "tasks_passed": passed_l3, "tasks_total": len(l3_details),
                "pass_rate": round(passed_l3 / len(l3_details), 2) if l3_details else 0,
                "avg_faithfulness": round(sum(r.faithfulness_score for r in l3_details) / len(l3_details), 2) if l3_details else 0,
                "avg_relevancy": round(sum(r.relevancy_score for r in l3_details) / len(l3_details), 2) if l3_details else 0,
                "details": [{"task_id": r.task_id, "faithfulness": r.faithfulness_score,
                              "relevancy": r.relevancy_score, "task_completion": r.task_completion_pass,
                              "fairness": r.fairness_pass, "overall": r.overall_pass}
                             for r in l3_details],
            })

        if run_l4:
            from eval_redteam import run_red_team, run_giskard_scan
            with st.spinner("Layer 4: Red-team probes..."):
                rt_result, probe_map = run_red_team(verbose=False)
                giskard = run_giskard_scan(probe_map)
            layer_results.append({
                "layer": 4, "name": "Red-Team",
                "total_probes": rt_result.total_probes,
                "defended": len(rt_result.passed),
                "critical_count": len(rt_result.critical_findings),
                "pass_rate": rt_result.overall_score,
                "giskard_scan": giskard,
                "critical_findings": [
                    {"probe_id": f.probe_id, "description": f.description}
                    for f in rt_result.critical_findings
                ],
            })

        if run_l5:
            from eval_governance import run_governance
            with st.spinner("Layer 5: Governance gate tests..."):
                gov_result = run_governance(verbose=False)
            layer_results.append({
                "layer": 5, "name": "Governance",
                "gate_coverage": gov_result.gate_coverage,
                "action_slipped": gov_result.action_slipped_count,
                "all_passed": gov_result.all_passed,
                "pass_rate": 1.0 if gov_result.all_passed and gov_result.action_slipped_count == 0 else 0.0,
                "tests": [{"test_id": t.test_id, "passed": t.passed, "note": t.note}
                          for t in gov_result.tests],
            })

        overall = sum(r["pass_rate"] for r in layer_results) / len(layer_results) if layer_results else 0
        st.session_state["eval_results"] = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "overall_score": round(overall, 2),
            "overall_status": "PASS" if overall >= 0.7 else "FAIL",
            "layers": layer_results,
        }
        st.rerun()

    if st.session_state["eval_results"]:
        results = st.session_state["eval_results"]
        overall = results["overall_score"]
        status  = results["overall_status"]

        st.divider()
        st.subheader(f"🏆 Scorecard — Overall: {overall:.0%}  ({status})")

        layer_cols = st.columns(len(results["layers"]))
        for i, lr in enumerate(results["layers"]):
            rate = lr["pass_rate"]
            layer_cols[i].metric(
                f"L{lr['layer']}: {lr['name']}",
                f"{rate:.0%}",
                delta=None,
            )

        st.divider()

        for lr in results["layers"]:
            lnum = lr["layer"]
            with st.expander(f"Layer {lnum} — {lr['name']}  ({lr['pass_rate']:.0%})", expanded=True):

                if lnum == 2:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Invariant pass rate", f"{lr.get('invariant_pass_rate', 0):.0%}")
                    c2.metric("Avg tool accuracy",   f"{lr.get('avg_tool_accuracy', 0):.0%}")
                    c3.metric("Avg judge score",     f"{lr.get('avg_judge_score', 0):.2f}")
                    for d in lr.get("details", []):
                        icon = "✅" if d["overall"] else "❌"
                        st.markdown(
                            f"{icon} **{d['task_id']}** — "
                            f"invariant: {'OK' if d['invariant_pass'] else 'FAIL'} | "
                            f"tool: {d['tool_accuracy']:.0%} | "
                            f"judge: {d['judge_score']:.2f}"
                        )
                        if d.get("failures"):
                            for fail in d["failures"]:
                                st.caption(f"    - {fail}")

                elif lnum == 3:
                    c1, c2 = st.columns(2)
                    c1.metric("Avg faithfulness", f"{lr.get('avg_faithfulness', 0):.2f}")
                    c2.metric("Avg relevancy",    f"{lr.get('avg_relevancy', 0):.2f}")
                    for d in lr.get("details", []):
                        icon = "✅" if d["overall"] else "❌"
                        st.markdown(
                            f"{icon} **{d['task_id']}** — "
                            f"faith: {d['faithfulness']:.2f} | "
                            f"relev: {d['relevancy']:.2f} | "
                            f"fair: {'OK' if d['fairness'] else 'FAIL'}"
                        )

                elif lnum == 4:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Probes defended", f"{lr.get('defended', 0)}/{lr.get('total_probes', 0)}")
                    c2.metric("Critical/High",   lr.get("critical_count", 0))
                    c3.metric("Overall score",   f"{lr.get('pass_rate', 0):.0%}")
                    if lr.get("critical_findings"):
                        st.error("Critical vulnerabilities found:")
                        for cf in lr["critical_findings"]:
                            st.markdown(f"- **[{cf['probe_id']}]** {cf['description']}")
                    else:
                        st.success("No critical vulnerabilities.")
                    giskard = lr.get("giskard_scan", {})
                    if giskard:
                        st.markdown("**Giskard-equivalent scan:**")
                        for cat, info in giskard.items():
                            icon = "✓" if info["status"] == "CLEAN" else "!"
                            color = "#0E8A5F" if info["status"] == "CLEAN" else "#D93636"
                            st.markdown(
                                f'<span style="color:{color}">[{icon}] {cat}</span>: '
                                f'{info["status"]} ({info["defended"]}/{info["total_probes"]} defended)',
                                unsafe_allow_html=True,
                            )

                elif lnum == 5:
                    c1, c2 = st.columns(2)
                    c1.metric("Gate coverage",    f"{lr.get('gate_coverage', 0):.0%}")
                    c2.metric("Actions slipped",  lr.get("action_slipped", 0))
                    if lr.get("action_slipped", 0) > 0:
                        st.error("CRITICAL: propose_interview fired without human approval!")
                    else:
                        st.success("No actions slipped through unapproved.")
                    for t in lr.get("tests", []):
                        icon = "✅" if t["passed"] else "❌"
                        st.markdown(f"{icon} **{t['test_id']}** — {t['note'][:90]}")

        st.divider()
        scorecard_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="⬇️ Download scorecard JSON",
            data=scorecard_json,
            file_name=f"scorecard_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("Clear evaluation results", use_container_width=True):
            st.session_state["eval_results"] = None
            st.rerun()