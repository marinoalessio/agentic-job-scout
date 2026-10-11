"""
Job Scout — Streamlit UI

Calls orchestrator and agent functions directly (no HTTP/FastAPI).
Uses st.session_state to preserve results across Streamlit reruns.
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.cv_parser import parse_cv
from agents.gap_analysis import analyse_gap
from agents.application_pack import generate_application_pack
from memory import user_prefs
from orchestrator import pipeline
from tracing import langfuse_handler

# ── Page config (must be the very first Streamlit call) ─────────────────────

st.set_page_config(
    page_title="Job Scout",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ────────────────────────────────────────────────────────────

FONTS = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap"

CSS = f"""
<style>
@import url('{FONTS}');

/* ── Base ── */
html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: #F7F6F2;
    color: #1A1A2E;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #EEECEA;
    border-right: 1px solid #DDDAD4;
}}
[data-testid="stSidebar"] * {{
    color: #3A3A4A !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #1A1A2E !important;
}}

/* ── Inputs (sidebar) ── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{
    background: #FAFAF8 !important;
    border: 1px solid #CCCAC4 !important;
    color: #1A1A2E !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {{
    border-color: #1B3A6B !important;
    box-shadow: 0 0 0 2px rgba(27, 58, 107, 0.12) !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {{
    background: #FAFAF8 !important;
    border: 1px dashed #BBBAAF !important;
    border-radius: 6px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: #1B3A6B !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: #FFFFFF;
    border: 1px solid #CCCAC4;
    color: #3A3A4A;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
    text-transform: uppercase;
}}
.stButton > button:hover {{
    background: #F0EEE8;
    border-color: #1B3A6B;
    color: #1B3A6B;
}}
.stButton > button:active {{
    background: #E8E4DC;
}}

/* Primary button (Search Jobs) */
.primary-btn .stButton > button {{
    background: #1B3A6B;
    border-color: #1B3A6B;
    color: #FFFFFF;
    width: 100%;
    padding: 0.6rem 1.2rem;
    font-size: 0.82rem;
}}
.primary-btn .stButton > button:hover {{
    background: #24508F;
    border-color: #24508F;
    color: #FFFFFF;
}}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {{
    padding: 5rem 2.5rem 2rem 2.5rem;
    max-width: 1100px;
}}

/* ── Typography ── */
h1 {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #1A1A2E; letter-spacing: -0.02em; }}
h2 {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #4A4A5A; letter-spacing: 0.02em; }}
h3 {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 1rem; font-weight: 600; color: #1A1A2E; }}

/* ── Dividers ── */
hr {{ border: none; border-top: 1px solid #DDDAD4; margin: 1.5rem 0; }}

/* ── Spinner ── */
[data-testid="stSpinner"] {{ color: #1B3A6B !important; }}

/* ── Success / info messages ── */
[data-testid="stAlert"] {{
    background: #EEF2F8 !important;
    border: 1px solid #C8D6EA !important;
    border-radius: 4px;
}}

/* ── Job card ── */
.job-card {{
    background: #FFFFFF;
    border: 1px solid #E4E1DA;
    border-left: 3px solid var(--score-color, #1B3A6B);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.card-header {{
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.75rem;
}}
.score-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
    color: var(--score-color);
    min-width: 2.5rem;
    padding-top: 0.1rem;
}}
.score-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #9A9AA8;
    letter-spacing: 0.1em;
    display: block;
    margin-top: 0.15rem;
}}
.job-title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #1A1A2E;
    margin: 0 0 0.3rem 0;
    line-height: 1.3;
}}
.job-meta {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}}
.source-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6A6A7A;
    background: #F2F0EA;
    border: 1px solid #DDDAD4;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    letter-spacing: 0.05em;
}}
.job-url {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #1B3A6B;
    text-decoration: none;
}}
.job-url:hover {{ color: #24508F; text-decoration: underline; }}
.job-rationale {{
    font-size: 0.85rem;
    color: #6A6A7A;
    line-height: 1.5;
    font-style: italic;
    border-left: 2px solid #E4E1DA;
    padding-left: 0.75rem;
    margin-top: 0.5rem;
}}

/* ── Gap analysis panel ── */
.gap-panel {{
    background: #FAFAF8;
    border: 1px solid #E4E1DA;
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 1rem 1.4rem;
    margin-top: -1px;
    margin-bottom: 0.4rem;
}}
.gap-section-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.75rem 0 0.3rem 0;
}}
.gap-item {{
    font-size: 0.84rem;
    line-height: 1.5;
    margin: 0.2rem 0;
    padding-left: 1rem;
    position: relative;
    color: #2A2A3A;
}}
.gap-item::before {{
    position: absolute;
    left: 0;
}}
.strength .gap-item::before {{ content: "+"; color: #15803D; }}
.critical  .gap-item::before {{ content: "✕"; color: #B91C1C; }}
.minor     .gap-item::before {{ content: "~"; color: #B45309; }}
.strength .gap-section-title {{ color: #15803D; }}
.critical  .gap-section-title {{ color: #B91C1C; }}
.minor     .gap-section-title {{ color: #B45309; }}
.recommendation-box {{
    background: #F0EEE8;
    border: 1px solid #DDDAD4;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #3A3A4A;
    line-height: 1.5;
}}
.recommendation-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #9A9AA8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}

/* ── Cover letter panel ── */
.cover-panel {{
    background: #FAFAF8;
    border: 1px solid #E4E1DA;
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 1rem 1.4rem;
    margin-top: -1px;
    margin-bottom: 0.4rem;
}}
.cover-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9A9AA8;
    margin-bottom: 0.5rem;
}}
.talking-point {{
    font-size: 0.84rem;
    line-height: 1.5;
    margin: 0.25rem 0;
    color: #3A3A4A;
    padding-left: 1rem;
    position: relative;
}}
.talking-point::before {{
    content: "›";
    position: absolute;
    left: 0;
    color: #1B3A6B;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 5rem 2rem;
    color: #BBBAAF;
}}
.empty-state-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
}}
.empty-state-text {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}}

/* ── Results header ── */
.results-header {{
    margin-bottom: 1.5rem;
}}
.results-count {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #1B3A6B;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}
.results-name {{
    font-size: 1.3rem;
    font-weight: 600;
    color: #1A1A2E;
}}
.results-location {{
    font-size: 0.85rem;
    color: #6A6A7A;
    margin-top: 0.1rem;
}}

/* ── CV status chip ── */
.cv-status {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.7rem;
    border-radius: 3px;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    display: inline-block;
}}
.cv-loaded {{
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #15803D;
}}
.cv-missing {{
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #B91C1C;
}}

/* ── Textarea (cover letter) ── */
.stTextArea textarea {{
    background: #FFFFFF !important;
    border: 1px solid #DDDAD4 !important;
    color: #2A2A3A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    border-radius: 4px !important;
}}

/* ── Top header bar ── */
[data-testid="stHeader"] {{
    background: #F7F6F2 !important;
    border-bottom: 1px solid #DDDAD4;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}
[data-testid="stToolbar"] {{
    background: #F7F6F2 !important;
}}

/* ── Sidebar logo ── */
.sidebar-logo {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1A1A2E;
    letter-spacing: 0.04em;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #DDDAD4;
    margin-bottom: 1.5rem;
}}
.sidebar-logo span {{
    color: #1B3A6B;
}}
.sidebar-section {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #9A9AA8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 1.25rem 0 0.5rem 0;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Session state initialization ─────────────────────────────────────────────

defaults = {
    "profile": None,
    "ranked": [],
    "gap_analyses": {},    # job url → GapAnalysis
    "cover_letters": {},   # job url → ApplicationPack
    "search_done": False,
    "location": user_prefs.get("target_location") or "",
    "cv_path": user_prefs.get("cv_path") or None,
    "cv_filename": user_prefs.get("cv_filename") or None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────────────

CV_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cv_uploads",
)
os.makedirs(CV_STORE_DIR, exist_ok=True)


def save_uploaded_cv(uploaded_file) -> str:
    """Saves the uploaded file to a persistent location. Returns the path."""
    dest = os.path.join(CV_STORE_DIR, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def score_color(score: int) -> str:
    if score >= 70:
        return "#22C55E"
    if score >= 50:
        return "#F59E0B"
    return "#EF4444"


def render_job_card(job, idx: int) -> None:
    color = score_color(job.score)
    # Truncate rationale if very long
    rationale = job.rationale if len(job.rationale) < 300 else job.rationale[:300] + "…"

    st.markdown(
        f"""
        <div class="job-card" style="--score-color: {color};">
            <div class="card-header">
                <div>
                    <div class="score-badge">{job.score}</div>
                    <span class="score-label">/100</span>
                </div>
                <div style="flex: 1; min-width: 0;">
                    <div class="job-title">{job.title}</div>
                    <div class="job-meta">
                        <span class="source-tag">{job.source}</span>
                        <a class="job-url" href="{job.url}" target="_blank">view posting →</a>
                    </div>
                </div>
            </div>
            <div class="job-rationale">{rationale}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gap_panel(gap) -> None:
    strengths_html = "\n".join(
        f'<div class="gap-item">{s}</div>' for s in gap.strengths
    )
    critical_html = "\n".join(
        f'<div class="gap-item">{c}</div>' for c in gap.critical_gaps
    )
    minor_html = "\n".join(
        f'<div class="gap-item">{m}</div>' for m in gap.minor_gaps
    )

    st.markdown(
        f"""
        <div class="gap-panel">
            <div class="strength">
                <div class="gap-section-title">Strengths</div>
                {strengths_html}
            </div>
            <div class="critical">
                <div class="gap-section-title">Critical gaps</div>
                {critical_html}
            </div>
            <div class="minor">
                <div class="gap-section-title">Minor gaps</div>
                {minor_html}
            </div>
            <div class="recommendation-box">
                <div class="recommendation-label">Verdict</div>
                {gap.recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cover_panel(pack) -> None:
    points_html = "\n".join(
        f'<div class="talking-point">{p}</div>' for p in pack.talking_points
    )
    st.markdown(
        f"""
        <div class="cover-panel">
            <div class="cover-label">Cover letter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_area(
        label="",
        value=pack.cover_letter,
        height=260,
        key=f"cl_text_{pack.job_title}",
        label_visibility="collapsed",
    )
    st.markdown(
        f"""
        <div style="margin-top: 0.75rem;">
            <div class="cover-label">Interview talking points</div>
            {points_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">◈ JOB <span>SCOUT</span></div>',
        unsafe_allow_html=True,
    )

    # ── CV section ──
    st.markdown('<div class="sidebar-section">Curriculum Vitae</div>', unsafe_allow_html=True)

    cv_path = st.session_state.cv_path
    cv_filename = st.session_state.cv_filename

    if cv_path and os.path.exists(cv_path):
        st.markdown(
            f'<div class="cv-status cv-loaded">◉ {cv_filename}</div>',
            unsafe_allow_html=True,
        )
        replace = st.checkbox("Replace CV", value=False, key="replace_cv")
        show_uploader = replace
    else:
        st.markdown(
            '<div class="cv-status cv-missing">○ No CV loaded</div>',
            unsafe_allow_html=True,
        )
        show_uploader = True

    if show_uploader:
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="cv_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            saved_path = save_uploaded_cv(uploaded)
            st.session_state.cv_path = saved_path
            st.session_state.cv_filename = uploaded.name
            user_prefs.set("cv_path", saved_path)
            user_prefs.set("cv_filename", uploaded.name)
            # Reset results when a new CV is loaded
            st.session_state.profile = None
            st.session_state.ranked = []
            st.session_state.gap_analyses = {}
            st.session_state.cover_letters = {}
            st.session_state.search_done = False
            st.rerun()

    # ── Location section ──
    st.markdown('<div class="sidebar-section">Target location</div>', unsafe_allow_html=True)

    location_input = st.text_input(
        "Location",
        value=st.session_state.location,
        placeholder="e.g. Sydney, Australia",
        key="location_input",
        label_visibility="collapsed",
    )
    if location_input != st.session_state.location:
        st.session_state.location = location_input
        user_prefs.set("target_location", location_input)

    # ── Results count ──
    st.markdown('<div class="sidebar-section">Number of results</div>', unsafe_allow_html=True)

    max_results = st.slider(
        "Number of results",
        min_value=3,
        max_value=10,
        value=5,
        key="max_results_slider",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Search button ──
    cv_ready = st.session_state.cv_path and os.path.exists(st.session_state.cv_path or "")
    loc_ready = bool(st.session_state.location.strip())
    can_search = cv_ready and loc_ready

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    search_clicked = st.button(
        "◈  SEARCH JOBS",
        key="search_btn",
        disabled=not can_search,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not can_search:
        if not cv_ready:
            st.caption("↑ Upload a CV to continue")
        elif not loc_ready:
            st.caption("↑ Enter a target location")

    # ── Footer ──
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family: IBM Plex Mono, monospace; font-size: 0.6rem; '
        'color: #BBBAAF; letter-spacing: 0.08em;">PHASE 5 · STREAMLIT UI</div>',
        unsafe_allow_html=True,
    )

# ── Search execution ─────────────────────────────────────────────────────────

if search_clicked and can_search:
    st.session_state.gap_analyses = {}
    st.session_state.cover_letters = {}
    st.session_state.search_done = False

    with st.spinner("Parsing CV…"):
        profile = parse_cv(st.session_state.cv_path)
        st.session_state.profile = profile

    with st.spinner(f"Searching jobs in {st.session_state.location}…"):
        final_state = pipeline.invoke(
            {
                "profile": profile,
                "location": st.session_state.location,
                "retried": False,
                "custom_query": "",
                "seen_urls": set(),
                "max_results": max_results,
            },
            config={"callbacks": [langfuse_handler]},
        )
        st.session_state.ranked = final_state["ranked"]
        st.session_state.search_done = True

    st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────

if not st.session_state.search_done:
    # Empty state
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">◈</div>
            <div class="empty-state-text">
                Upload your CV · set a location<br>
                then hit SEARCH JOBS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    profile = st.session_state.profile
    ranked = st.session_state.ranked

    # Results header
    st.markdown(
        f"""
        <div class="results-header">
            <div class="results-count">◈ {len(ranked)} jobs found</div>
            <div class="results-name">{profile.name}</div>
            <div class="results-location">{st.session_state.location}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Job cards ──
    for idx, job in enumerate(ranked):
        render_job_card(job, idx)

        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            gap_done = job.url in st.session_state.gap_analyses
            gap_label = "Gap ✓" if gap_done else "Gap Analysis"
            if st.button(gap_label, key=f"gap_btn_{idx}"):
                if not gap_done:
                    with st.spinner(f'Analysing fit for "{job.title}"...'):
                        gap = analyse_gap(profile, job)
                        st.session_state.gap_analyses[job.url] = gap
                    st.rerun()

        with col2:
            cover_done = job.url in st.session_state.cover_letters
            cover_label = "Letter ✓" if cover_done else "Cover Letter"
            if st.button(cover_label, key=f"cover_btn_{idx}"):
                if not cover_done:
                    # Reuse gap if already computed, else compute it
                    if job.url in st.session_state.gap_analyses:
                        gap = st.session_state.gap_analyses[job.url]
                    else:
                        with st.spinner("Running gap analysis first…"):
                            gap = analyse_gap(profile, job)
                            st.session_state.gap_analyses[job.url] = gap
                    with st.spinner(f'Writing cover letter for "{job.title}"...'):
                        pack = generate_application_pack(profile, job, gap)
                        st.session_state.cover_letters[job.url] = pack
                    st.rerun()

        # Inline gap analysis panel
        if job.url in st.session_state.gap_analyses:
            render_gap_panel(st.session_state.gap_analyses[job.url])

        # Inline cover letter panel
        if job.url in st.session_state.cover_letters:
            render_cover_panel(st.session_state.cover_letters[job.url])

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
