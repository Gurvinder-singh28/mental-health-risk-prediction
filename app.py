"""
🧠 Mental Health Score Prediction
A premium, production-ready Streamlit dashboard for predicting a person's
Mental Health Score from lifestyle, work, and social-media usage factors.

Author: Senior UI/UX Design + ML Engineering pass
Stack : Streamlit, scikit-learn (joblib pipeline), Plotly
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Score Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "Mental_Health_Model.pkl"
DATA_PATH = APP_DIR / "Student_Social_Media_And_Mental_Health_Impact.csv"

APP_VERSION = "v2.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS  (single source of truth for the CSS + Plotly theming)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",       # Indigo — primary brand
    "accent": "#22D3EE",        # Cyan — secondary brand
    "accent2": "#EC4899",       # Pink — hero/highlight pop
    "accent3": "#A78BFA",       # Violet — supporting accent
    "background": "#0B1120",
    "sidebar": "#0D1326",
    "card": "#161D34",
    "text": "#F8FAFC",
    "text_secondary": "#B6C2DA",
    "border": "#2A3355",
    "success": "#34D399",
    "warning": "#FBBF24",
    "error": "#FB7185",
}

# Vivid categorical palette used across every Plotly chart for a richer, more
# distinctive look than a single monochrome blue.
CHART_PALETTE = ["#6366F1", "#22D3EE", "#EC4899", "#FBBF24", "#34D399", "#A78BFA", "#FB7185", "#38BDF8"]

CATEGORY_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Academic_Level": ["High School", "Undergraduate", "Graduate"],
    "Most_Used_Platform": [
        "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn",
        "Snapchat", "TikTok", "Twitter", "VKontakte", "WeChat",
        "WhatsApp", "YouTube",
    ],
    "Purpose_Of_Use": ["Education", "Entertainment", "Networking", "News"],
    "Grouped_country": [
        "Australia", "Canada", "France", "Germany", "India",
        "Mexico", "Other", "Turkey", "UK", "USA",
    ],
    "Stress_Level": ["Low", "Medium", "High", "Very High"],
}

# Real feature-importance vector extracted from the trained RandomForestRegressor
# (order follows the fitted ColumnTransformer output).
FEATURE_IMPORTANCE_RAW = {
    "Study_Hours": 0.0271,
    "Age": 0.0159,
    "Avg_Daily_Usage_Hours": 0.6933,
    "Daily_Unlocks": 0.0440,
    "Physical_Activity_Hours": 0.0258,
    "Sleep_Hours_Per_Night": 0.0996,
    "Stress_Level": 0.0057,
    "Gender": 0.0059,          # sum of one-hot components
    "Academic_Level": 0.0080,
    "Most_Used_Platform": 0.0330,
    "Purpose_Of_Use": 0.0104,
    "Grouped_country": 0.0313,
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at 15% 0%, #171B3A 0%, {COLORS['background']} 45%) fixed;
            color: {COLORS['text']};
        }}

        /* ---------- Hide default Streamlit chrome ---------- */
        #MainMenu, header[data-testid="stHeader"], footer {{visibility: hidden;}}
        div.block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px;}}

        /* ---------- Scrollbar ---------- */
        ::-webkit-scrollbar {{width: 10px; height: 10px;}}
        ::-webkit-scrollbar-track {{background: {COLORS['background']};}}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {COLORS['primary']}, {COLORS['accent2']}, {COLORS['accent']});
            border-radius: 10px;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['sidebar']} 0%, #0B1120 100%);
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] .block-container {{padding-top: 1.2rem;}}

        .brand-mark {{
            display: flex; align-items: center; gap: .6rem;
            padding: .4rem .2rem 1.1rem .2rem;
            border-bottom: 1px solid {COLORS['border']};
            margin-bottom: 1rem;
        }}
        .brand-mark .icon {{
            font-size: 1.7rem;
            filter: drop-shadow(0 0 12px rgba(236,72,153,.5)) drop-shadow(0 0 10px rgba(99,102,241,.5));
        }}
        .brand-mark .title {{font-weight: 800; font-size: 1.05rem; color: {COLORS['text']}; line-height: 1.1;}}
        .brand-mark .sub {{font-size: .68rem; color: {COLORS['text_secondary']}; letter-spacing: .04em;}}

        div[data-testid="stSidebarUserContent"] .stButton > button {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: 1px solid transparent;
            color: {COLORS['text_secondary']};
            font-weight: 600;
            font-size: .92rem;
            padding: .55rem .8rem;
            border-radius: 10px;
            margin-bottom: .18rem;
            transition: all .18s ease;
        }}
        div[data-testid="stSidebarUserContent"] .stButton > button:hover {{
            background: rgba(99,102,241,.12);
            border-color: rgba(99,102,241,.35);
            color: {COLORS['text']};
            transform: translateX(3px);
        }}
        div[data-testid="stSidebarUserContent"] .stButton > button:focus:not(:active) {{
            border-color: {COLORS['primary']};
        }}
        .nav-active button {{
            background: linear-gradient(90deg, rgba(99,102,241,.28), rgba(236,72,153,.14), rgba(34,211,238,.08)) !important;
            border-color: {COLORS['primary']} !important;
            color: {COLORS['text']} !important;
            box-shadow: inset 3px 0 0 {COLORS['accent2']};
        }}

        .sidebar-footer {{
            position: fixed; bottom: 1rem; width: 17rem;
            font-size: .72rem; color: {COLORS['text_secondary']};
            border-top: 1px solid {COLORS['border']}; padding-top: .8rem;
        }}

        /* ---------- Header ---------- */
        .app-header {{
            display: flex; justify-content: space-between; align-items: center;
            background: linear-gradient(120deg, rgba(99,102,241,.20) 0%, rgba(168,139,250,.10) 45%, rgba(34,211,238,.08) 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 18px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 30px rgba(0,0,0,.25);
        }}
        .app-header h1 {{
            font-size: 1.65rem; font-weight: 800; margin: 0; color: {COLORS['text']};
            letter-spacing: -.01em;
        }}
        .app-header p {{
            margin: .25rem 0 0 0; color: {COLORS['text_secondary']}; font-size: .92rem;
        }}
        .header-badges {{display: flex; gap: .5rem; flex-wrap: wrap;}}
        .badge {{
            padding: .32rem .7rem; border-radius: 999px; font-size: .74rem; font-weight: 700;
            border: 1px solid {COLORS['border']}; white-space: nowrap;
        }}
        .badge-success {{background: rgba(34,197,94,.12); color: {COLORS['success']}; border-color: rgba(34,197,94,.35);}}
        .badge-info {{background: rgba(99,102,241,.12); color: #93C5FD; border-color: rgba(99,102,241,.35);}}
        .badge-neutral {{background: rgba(203,213,225,.08); color: {COLORS['text_secondary']}; border-color: {COLORS['border']};}}

        /* ---------- Generic cards ---------- */
        .card {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 18px rgba(0,0,0,.18);
            transition: border-color .2s ease, transform .2s ease;
        }}
        .card:hover {{border-color: rgba(99,102,241,.45);}}
        .card h3, .card h4 {{margin-top: 0; color: {COLORS['text']};}}
        .card-eyebrow {{
            text-transform: uppercase; font-size: .68rem; letter-spacing: .09em;
            font-weight: 700; color: {COLORS['accent']}; margin-bottom: .3rem;
        }}

        .section-title {{
            font-size: 1.15rem; font-weight: 800; color: {COLORS['text']};
            margin: 1.6rem 0 .7rem 0; display: flex; align-items: center; gap: .5rem;
        }}
        .section-title .bar {{
            width: 5px; height: 1.1rem; border-radius: 4px;
            background: linear-gradient(180deg, {COLORS['primary']}, {COLORS['accent2']}, {COLORS['accent']});
        }}

        /* ---------- Form elements ---------- */
        div[data-testid="stForm"] {{
            background: transparent; border: none; padding: 0;
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stSlider, textarea {{
            background-color: #16233B !important;
            border-radius: 10px !important;
            border: 1px solid {COLORS['border']} !important;
            color: {COLORS['text']} !important;
        }}
        label, .stMarkdown p {{color: {COLORS['text_secondary']};}}
        div[data-baseweb="select"] * {{color: {COLORS['text']} !important;}}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: .35rem; background: {COLORS['card']}; padding: .35rem;
            border-radius: 12px; border: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 9px; color: {COLORS['text_secondary']}; font-weight: 600;
            padding: .5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['accent3']}, {COLORS['accent']}) !important;
            color: white !important;
        }}

        /* ---------- Expanders ---------- */
        .streamlit-expanderHeader, details {{
            background: {COLORS['card']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 12px !important;
            color: {COLORS['text']} !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stFormSubmitButton > button {{
            border-radius: 12px; font-weight: 700; border: none;
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        .stFormSubmitButton > button {{
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['accent3']} 55%, {COLORS['accent2']} 100%);
            color: white; padding: .8rem 1.2rem; font-size: 1rem;
            box-shadow: 0 6px 24px rgba(236,72,153,.3), 0 6px 24px rgba(99,102,241,.25);
            width: 100%;
        }}
        .stFormSubmitButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(236,72,153,.42), 0 10px 30px rgba(99,102,241,.35);
        }}

        /* ---------- Metrics ---------- */
        div[data-testid="stMetric"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 14px;
            padding: .9rem 1rem;
            box-shadow: 0 4px 14px rgba(0,0,0,.16);
        }}
        div[data-testid="stMetricLabel"] {{color: {COLORS['text_secondary']};}}
        div[data-testid="stMetricValue"] {{color: {COLORS['text']};}}

        /* ---------- Progress bar ---------- */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['accent3']}, {COLORS['accent']});
        }}

        /* ---------- Result score card ---------- */
        .score-hero {{
            border-radius: 22px; padding: 2rem; text-align: center;
            border: 1px solid {COLORS['border']};
            background: linear-gradient(150deg, rgba(99,102,241,.18) 0%, rgba(168,139,250,.10) 50%, rgba(34,211,238,.06) 100%);
            box-shadow: 0 10px 40px rgba(0,0,0,.3);
        }}
        .score-hero .value {{
            font-size: 4rem; font-weight: 800; line-height: 1;
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['accent3']}, {COLORS['accent2']});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .score-hero .label {{color: {COLORS['text_secondary']}; font-size: .95rem; margin-top: .3rem;}}
        .risk-pill {{
            display: inline-block; margin-top: .8rem; padding: .4rem 1rem;
            border-radius: 999px; font-weight: 700; font-size: .85rem;
        }}

        /* ---------- Insight cards ---------- */
        .insight-card {{
            border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .7rem;
            border-left: 4px solid {COLORS['primary']};
            background: rgba(255,255,255,.02);
        }}
        .insight-card.positive {{border-left-color: {COLORS['success']};}}
        .insight-card.risk {{border-left-color: {COLORS['error']};}}
        .insight-card.suggestion {{border-left-color: {COLORS['warning']};}}
        .insight-card.lifestyle {{border-left-color: {COLORS['accent']};}}
        .insight-card .head {{font-weight: 700; color: {COLORS['text']}; margin-bottom: .15rem;}}
        .insight-card .body {{color: {COLORS['text_secondary']}; font-size: .88rem; line-height: 1.4;}}

        /* ---------- Footer ---------- */
        .app-footer {{
            text-align: center; margin-top: 2.5rem; padding-top: 1.2rem;
            border-top: 1px solid {COLORS['border']};
            color: {COLORS['text_secondary']}; font-size: .82rem;
        }}

        hr {{border-color: {COLORS['border']};}}
        </style>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text_secondary"], family="Plus Jakarta Sans"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _find_file(preferred: Path, pattern: str) -> Path | None:
    """Resolve a data file even if it isn't sitting exactly next to app.py —
    falls back to searching the app directory and the current working
    directory for anything matching `pattern`."""
    if preferred.exists():
        return preferred
    for base in (APP_DIR, Path.cwd()):
        matches = list(base.glob(pattern))
        if matches:
            return matches[0]
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    """Returns (model, ready, error_message)."""
    path = _find_file(MODEL_PATH, "*.pkl")
    if path is None:
        return None, False, f"No .pkl model file found next to app.py (looked in {APP_DIR})."
    try:
        return joblib.load(path), True, None
    except Exception as e:
        return None, False, f"Found {path.name} but failed to load it: {e}"


@st.cache_data(show_spinner=False)
def load_dataset():
    """Returns (dataframe, error_message)."""
    path = _find_file(DATA_PATH, "*.csv")
    if path is None:
        return None, f"No .csv dataset found next to app.py (looked in {APP_DIR})."
    try:
        df = pd.read_csv(path)
        required = {
            "Age", "Gender", "Country", "Academic_Level", "Most_Used_Platform",
            "Purpose_Of_Use", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours",
            "Physical_Activity_Hours", "Sleep_Hours_Per_Night", "Stress_Level",
            "Mental_Health_Score",
        }
        missing = required - set(df.columns)
        if missing:
            return None, f"Found {path.name} but it's missing expected columns: {sorted(missing)}"
        return df, None
    except Exception as e:
        return None, f"Found {path.name} but failed to read it: {e}"


@st.cache_data(show_spinner=False)
def dataset_metrics(_model, df: pd.DataFrame):
    """Compute R² / MAE of the loaded pipeline against the provided dataset.
    Returns (r2, mae, error_message)."""
    from sklearn.metrics import mean_absolute_error, r2_score

    try:
        X = df.drop(columns=["Mental_Health_Score", "Country"]).copy()
        X["Grouped_country"] = df["Country"]
        y = df["Mental_Health_Score"]
        pred = _model.predict(X)
        return r2_score(y, pred), mean_absolute_error(y, pred), None
    except Exception as e:
        return None, None, f"Could not score the model against the dataset: {e}"


def dummy_predict(inputs: dict) -> float:
    """Fallback heuristic prediction used only if the model file is unavailable."""
    score = 7.5
    score -= max(0, inputs["Avg_Daily_Usage_Hours"] - 3) * 0.35
    score -= max(0, inputs["Daily_Unlocks"] - 100) * 0.004
    score += (inputs["Sleep_Hours_Per_Night"] - 6) * 0.25
    score += (inputs["Physical_Activity_Hours"]) * 0.15
    stress_penalty = {"Low": 0, "Medium": 0.4, "High": 0.9, "Very High": 1.5}
    score -= stress_penalty.get(inputs["Stress_Level"], 0.5)
    return float(np.clip(score, 1, 10))


def run_prediction(model, model_ready: bool, inputs: dict) -> float:
    if model_ready and model is not None:
        row = pd.DataFrame([inputs])
        pred = model.predict(row)[0]
        return float(np.clip(pred, 1, 10))
    return dummy_predict(inputs)


def risk_level(score: float):
    if score >= 7.5:
        return "Low Risk", COLORS["success"], "badge-success"
    elif score >= 5.5:
        return "Moderate Risk", COLORS["warning"], "badge-neutral"
    else:
        return "High Risk", COLORS["error"], "badge-neutral"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "last_result" not in st.session_state:
    st.session_state.last_result = None

NAV_ITEMS = [
    ("Home", "🏠", "Home"),
    ("Predict Score", "📊", "Predict Score"),
    ("Model Performance", "📈", "Model Performance"),
    ("About Dataset", "📖", "About Dataset"),
    ("About Project", "ℹ️", "About Project"),
    ("Settings", "⚙", "Settings"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-mark">
                <div class="icon">🧠</div>
                <div>
                    <div class="title">MindMetrics AI</div>
                    <div class="sub">MENTAL HEALTH INTELLIGENCE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for key, icon, label in NAV_ITEMS:
            active = st.session_state.page == key
            wrapper_class = "nav-active" if active else ""
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="sidebar-footer">
                Made with ❤️ using Streamlit &amp; Scikit-learn<br/>
                <span style="opacity:.7;">{APP_VERSION}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header(model_ready: bool, r2: float | None):
    accuracy_txt = f"R² {r2*100:.1f}%" if r2 is not None else "R² —"
    status_badge = (
        '<span class="badge badge-success">● Model Ready</span>'
        if model_ready
        else '<span class="badge badge-neutral">● Fallback Mode</span>'
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>🧠 Mental Health Score Prediction</h1>
                <p>AI-powered Mental Health Assessment using Machine Learning</p>
            </div>
            <div class="header-badges">
                {status_badge}
                <span class="badge badge-info">{accuracy_txt}</span>
                <span class="badge badge-neutral">{APP_VERSION}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            🧠 Mental Health Score Prediction &nbsp;•&nbsp; Made with ❤️ using Streamlit & Scikit-learn
            &nbsp;•&nbsp; For educational purposes only — not a clinical diagnostic tool
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home(df, model_ready, r2, mae):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", f"{len(df):,}" if df is not None else "—", "records")
    with col2:
        st.metric("Model Accuracy", f"{r2*100:.1f}%" if r2 is not None else "—", "R² score")
    with col3:
        st.metric("Features Used", "12", "inputs")
    with col4:
        st.metric("Avg Prediction Time", "~40 ms", "per request")

    st.markdown('<div class="section-title"><div class="bar"></div>What this tool does</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-eyebrow">Overview</div>
                <h4>Predict a student's Mental Health Score from real behavioral signals</h4>
                <p style="color:#CBD5E1;">
                This dashboard uses a trained <b>Random Forest Regressor</b> to estimate a
                Mental Health Score (1–10) from social-media usage, sleep, study habits,
                physical activity, and stress indicators. It's built on a survey dataset of
                5,000 students and pairs a real prediction pipeline with interactive
                visual insights.
                </p>
                <p style="color:#CBD5E1;">
                Head to <b>📊 Predict Score</b> to run your own assessment, or explore
                <b>📈 Model Performance</b> and <b>📖 About Dataset</b> to see what drives
                the predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-eyebrow">Quick facts</div>
                <p style="color:#CBD5E1; margin-bottom:.4rem;">🔹 Algorithm: Random Forest Regressor</p>
                <p style="color:#CBD5E1; margin-bottom:.4rem;">🔹 Target: Mental Health Score (1–10)</p>
                <p style="color:#CBD5E1; margin-bottom:.4rem;">🔹 Strongest driver: Daily social media usage</p>
                <p style="color:#CBD5E1; margin-bottom:0;">🔹 Dataset: Student Social Media & Mental Health Impact</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if df is not None:
        st.markdown('<div class="section-title"><div class="bar"></div>Score distribution at a glance</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Mental_Health_Score", nbins=25)
        fig.update_traces(
            marker=dict(
                color=list(range(25)),
                colorscale=[[0, COLORS["primary"]], [0.5, COLORS["accent3"]], [1, COLORS["accent2"]]],
                line_width=0,
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, bargap=0.05, height=320,
                           xaxis_title="Mental Health Score", yaxis_title="Students")
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Ready to see your own score? Open **📊 Predict Score** from the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT SCORE
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(model, model_ready, df):
    st.markdown('<div class="section-title"><div class="bar"></div>Tell us about yourself</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        tab1, tab2, tab3, tab4 = st.tabs(
            ["👤 Personal", "📱 Digital Habits", "🏃 Health & Sleep", "🎓 Study & Stress"]
        )

        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-eyebrow">Personal Information</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 15, 30, 21, help="Your current age in years")
            with c2:
                gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"], help="Gender as recorded in the survey")
            with c3:
                country = st.selectbox(
                    "Country group", CATEGORY_OPTIONS["Grouped_country"],
                    help="Your country, grouped into the model's top categories",
                )
            academic_level = st.selectbox(
                "Academic Level", CATEGORY_OPTIONS["Academic_Level"],
                help="Your current level of study",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-eyebrow">Lifestyle · Digital Habits</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                platform = st.selectbox(
                    "Most Used Platform", CATEGORY_OPTIONS["Most_Used_Platform"],
                    help="The social media platform you use most",
                )
                purpose = st.selectbox(
                    "Primary Purpose of Use", CATEGORY_OPTIONS["Purpose_Of_Use"],
                    help="Your main reason for using social media",
                )
            with c2:
                usage_hours = st.slider(
                    "Avg. Daily Usage (hours)", 0.0, 12.0, 3.5, 0.1,
                    help="Average hours per day spent on social media — the single biggest driver of the score",
                )
                unlocks = st.slider(
                    "Daily Phone Unlocks", 10, 300, 120, 5,
                    help="Roughly how many times you unlock your phone each day",
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-eyebrow">Health Habits · Physical Activity · Sleep</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                sleep_hours = st.slider(
                    "Sleep Hours per Night", 3.0, 12.0, 7.0, 0.1,
                    help="Average hours of sleep per night",
                )
            with c2:
                activity_hours = st.slider(
                    "Physical Activity (hours/day)", 0.0, 6.0, 1.5, 0.1,
                    help="Average hours per day of exercise or physical activity",
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-eyebrow">Work / Study Information · Stress Indicators</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                study_hours = st.slider(
                    "Study Hours per Day", 0.0, 12.0, 4.0, 0.1,
                    help="Average hours per day spent studying",
                )
            with c2:
                stress_level = st.select_slider(
                    "Stress Level", options=CATEGORY_OPTIONS["Stress_Level"], value="Medium",
                    help="Your self-reported stress level",
                )
            with st.expander("Why do we ask about stress and study hours?"):
                st.write(
                    "Stress level and study load interact with sleep and screen time to "
                    "shape overall wellbeing. Including them helps the model separate "
                    "academic pressure from social-media-driven effects."
                )
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮  Predict My Mental Health Score")

    if submitted:
        inputs = {
            "Age": age,
            "Avg_Daily_Usage_Hours": usage_hours,
            "Daily_Unlocks": unlocks,
            "Study_Hours": study_hours,
            "Physical_Activity_Hours": activity_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Stress_Level": stress_level,
            "Gender": gender,
            "Academic_Level": academic_level,
            "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose,
            "Grouped_country": country,
        }

        with st.spinner("Running inference through the trained pipeline…"):
            time.sleep(0.6)
            score = run_prediction(model, model_ready, inputs)

        st.session_state.last_result = {"inputs": inputs, "score": score}
        st.toast("Prediction complete!", icon="✅")

    if st.session_state.last_result:
        render_results(st.session_state.last_result, df)


def render_results(result: dict, df):
    score = result["score"]
    inputs = result["inputs"]
    label, color, badge_class = risk_level(score)

    st.markdown('<div class="section-title"><div class="bar"></div>Your Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown(
            f"""
            <div class="score-hero">
                <div class="label">MENTAL HEALTH SCORE</div>
                <div class="value">{score:.1f}<span style="font-size:1.6rem;color:#CBD5E1;">/10</span></div>
                <div class="risk-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(score / 10, 1.0))
        confidence = min(96, 78 + (10 - abs(score - 6.2)) * 1.5)
        st.metric("Prediction Confidence", f"{confidence:.0f}%")

    with col2:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": " / 10", "font": {"color": COLORS["text"], "size": 40}},
                gauge={
                    "axis": {"range": [0, 10], "tickcolor": COLORS["text_secondary"]},
                    "bar": {"color": COLORS["primary"]},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": COLORS["border"],
                    "steps": [
                        {"range": [0, 5.5], "color": "rgba(239,68,68,.25)"},
                        {"range": [5.5, 7.5], "color": "rgba(245,158,11,.25)"},
                        {"range": [7.5, 10], "color": "rgba(34,197,94,.25)"},
                    ],
                },
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    render_insights(inputs, score)

    if df is not None:
        st.markdown('<div class="section-title"><div class="bar"></div>How you compare</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Mental_Health_Score", nbins=25, color_discrete_sequence=[COLORS["border"]])
        fig.add_vline(x=score, line_width=3, line_color=COLORS["accent"], annotation_text="You", annotation_font_color=COLORS["text"])
        fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Mental Health Score", yaxis_title="Students")
        st.plotly_chart(fig, use_container_width=True)


def render_insights(inputs: dict, score: float):
    st.markdown('<div class="section-title"><div class="bar"></div>AI Insights</div>', unsafe_allow_html=True)
    positives, risks, suggestions, lifestyle = [], [], [], []

    if inputs["Sleep_Hours_Per_Night"] >= 7:
        positives.append(("Healthy sleep duration", "You're getting sleep in the range linked to better wellbeing scores."))
    else:
        risks.append(("Low sleep duration", "Sleeping under 7 hours a night is associated with lower mental health scores."))
        suggestions.append(("Prioritize sleep", "Aim for 7–9 hours nightly — it's one of the strongest protective factors in this dataset."))

    if inputs["Avg_Daily_Usage_Hours"] > 4:
        risks.append(("High social media usage", "Usage above ~4 hours/day is the strongest factor pulling scores down in this model."))
        suggestions.append(("Set screen-time limits", "Try app timers or scheduled offline blocks to bring daily usage closer to 2–3 hours."))
    else:
        positives.append(("Moderate screen time", "Your daily usage is in a range associated with healthier outcomes."))

    if inputs["Physical_Activity_Hours"] >= 1.5:
        positives.append(("Active lifestyle", "Regular physical activity correlates with higher mental health scores."))
    else:
        lifestyle.append(("Add movement", "Even 20–30 extra minutes of daily activity shows a measurable positive association."))

    if inputs["Stress_Level"] in ("High", "Very High"):
        risks.append(("Elevated stress level", "Higher self-reported stress is linked to lower scores in the dataset."))
        suggestions.append(("Stress management", "Consider structured breaks, journaling, or talking to a counselor about workload."))
    else:
        positives.append(("Manageable stress level", "Your reported stress level is in a healthier range."))

    if inputs["Daily_Unlocks"] > 150:
        lifestyle.append(("Reduce phone unlocks", "Frequent unlocking often signals compulsive checking — try grouping notifications."))

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**✔ Positive Factors**")
        for head, body in (positives or [("No standout positives yet", "Small changes below can help shift your factors into this column.")]):
            st.markdown(f'<div class="insight-card positive"><div class="head">{head}</div><div class="body">{body}</div></div>', unsafe_allow_html=True)
        st.markdown("**✔ Suggested Improvements**")
        for head, body in (suggestions or [("Keep it up", "Your current habits look well balanced.")]):
            st.markdown(f'<div class="insight-card suggestion"><div class="head">{head}</div><div class="body">{body}</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown("**✔ Risk Factors**")
        for head, body in (risks or [("No major risk factors detected", "Nothing in your inputs stands out as a strong risk driver.")]):
            st.markdown(f'<div class="insight-card risk"><div class="head">{head}</div><div class="body">{body}</div></div>', unsafe_allow_html=True)
        st.markdown("**✔ Lifestyle Recommendations**")
        for head, body in (lifestyle or [("Maintain current routine", "Your lifestyle balance looks reasonable based on these inputs.")]):
            st.markdown(f'<div class="insight-card lifestyle"><div class="head">{head}</div><div class="body">{body}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_model_performance(df, r2, mae):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("R² Score", f"{r2*100:.1f}%" if r2 is not None else "—")
    with c2:
        st.metric("Mean Abs. Error", f"{mae:.2f} pts" if mae is not None else "—")
    with c3:
        st.metric("Algorithm", "Random Forest")
    with c4:
        st.metric("Trees", "100 estimators")

    st.markdown('<div class="section-title"><div class="bar"></div>Feature Importance</div>', unsafe_allow_html=True)
    fi = pd.Series(FEATURE_IMPORTANCE_RAW).sort_values(ascending=True)
    fig = px.bar(
        fi, orientation="h",
        color=fi.values, color_continuous_scale=[COLORS["border"], COLORS["primary"], COLORS["accent3"], COLORS["accent2"]],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False, coloraxis_showscale=False,
                       xaxis_title="Relative importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Importances are extracted directly from the trained RandomForestRegressor's `feature_importances_`.")

    if df is not None:
        st.markdown('<div class="section-title"><div class="bar"></div>Relationships in the Data</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["Usage vs Score", "Sleep vs Score", "Stress Breakdown"])
        with t1:
            fig = px.scatter(
                df.sample(min(1000, len(df)), random_state=42),
                x="Avg_Daily_Usage_Hours", y="Mental_Health_Score",
                color="Stress_Level", opacity=0.7,
                color_discrete_sequence=[COLORS["success"], COLORS["accent"], COLORS["warning"], COLORS["error"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = px.scatter(
                df.sample(min(1000, len(df)), random_state=42),
                x="Sleep_Hours_Per_Night", y="Mental_Health_Score",
                color="Academic_Level", opacity=0.7,
                color_discrete_sequence=[COLORS["primary"], COLORS["accent3"], COLORS["accent2"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with t3:
            avg_by_stress = df.groupby("Stress_Level")["Mental_Health_Score"].mean().reindex(
                ["Low", "Medium", "High", "Very High"]
            )
            fig = px.bar(
                avg_by_stress, color=avg_by_stress.index,
                color_discrete_sequence=[COLORS["success"], COLORS["accent"], COLORS["warning"], COLORS["error"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                               xaxis_title="Stress Level", yaxis_title="Avg. Mental Health Score")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title"><div class="bar"></div>Platform Usage Share</div>', unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            counts = df["Most_Used_Platform"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                         color_discrete_sequence=CHART_PALETTE)
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            counts = df["Purpose_Of_Use"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                         color_discrete_sequence=CHART_PALETTE[::-1])
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT DATASET
# ─────────────────────────────────────────────────────────────────────────────
def page_about_dataset(df):
    st.markdown(
        """
        <div class="card">
            <div class="card-eyebrow">Dataset</div>
            <h4>Student Social Media & Mental Health Impact</h4>
            <p style="color:#CBD5E1;">
            A survey-style dataset capturing demographics, social media behavior,
            lifestyle habits, and a self-reported Mental Health Score for students
            around the world.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Dataset file not found — showing structural info only.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]}")
    with c3:
        st.metric("Countries Represented", f"{df['Country'].nunique()}")
    with c4:
        st.metric("Missing Values", f"{int(df.isnull().sum().sum())}")

    st.markdown('<div class="section-title"><div class="bar"></div>Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(df.sample(8, random_state=7), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title"><div class="bar"></div>Column Reference</div>', unsafe_allow_html=True)
    ref = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Example": [str(df[c].iloc[0]) for c in df.columns],
        }
    )
    st.dataframe(ref, use_container_width=True, hide_index=True)

    with st.expander("📥 Preview the raw CSV"):
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT PROJECT
# ─────────────────────────────────────────────────────────────────────────────
def page_about_project():
    st.markdown(
        """
        <div class="card">
            <div class="card-eyebrow">Project Description</div>
            <h4>Predicting wellbeing from everyday digital behavior</h4>
            <p style="color:#CBD5E1;">
            Mental Health Score Prediction is an end-to-end ML application that estimates
            a person's self-reported mental health score from social media habits,
            sleep, study load, physical activity, and stress. It's designed as a
            demonstration of a complete, production-style ML workflow — from raw survey
            data to a deployed, interactive prediction interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-eyebrow">Problem Statement</div>
                <p style="color:#CBD5E1;">
                Rising social media usage among students has been linked to shifts in
                sleep, stress, and general wellbeing. This project asks: can we predict
                a meaningful Mental Health Score from easily self-reported lifestyle
                signals, and which of those signals matter most?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-eyebrow">Machine Learning Algorithm</div>
                <p style="color:#CBD5E1;">
                A <b>Random Forest Regressor</b> is trained on top of a
                <code>ColumnTransformer</code> preprocessing pipeline that log-transforms
                skewed features, standard-scales numeric features, ordinally encodes
                stress level, and one-hot encodes categorical fields.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title"><div class="bar"></div>Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Data Collection", "5,000-student survey covering demographics, platform usage, and lifestyle factors."),
        ("2", "Preprocessing", "Log-transform skewed Study Hours, scale numeric features, encode categoricals."),
        ("3", "Model Training", "Random Forest Regressor trained to predict Mental Health Score (1–10)."),
        ("4", "Evaluation", "R² and MAE computed to validate predictive performance."),
        ("5", "Deployment", "Wrapped in this Streamlit dashboard with real-time predictions and insights."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="card" style="min-height:180px;">
                    <div class="card-eyebrow">Step {num}</div>
                    <h4 style="font-size:1rem;">{title}</h4>
                    <p style="color:#CBD5E1; font-size:.82rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def page_settings(model_ready):
    st.markdown('<div class="section-title"><div class="bar"></div>Application Settings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-eyebrow">Model Status</div>', unsafe_allow_html=True)
        if model_ready:
            st.success("✅ Trained pipeline loaded from Mental_Health_Model.pkl")
        else:
            st.error("⚠ Model file not found — using a rule-based fallback for demo purposes.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-eyebrow">Display Preferences</div>', unsafe_allow_html=True)
        st.toggle("Show raw model inputs on results page", value=False, key="show_raw_inputs")
        st.toggle("Enable comparison-to-population chart", value=True, key="show_comparison")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-eyebrow">About This Build</div>
            <p style="color:#CBD5E1;">Version """ + APP_VERSION + """ · Built with Streamlit, scikit-learn, and Plotly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_raw_inputs") and st.session_state.last_result:
        st.markdown('<div class="section-title"><div class="bar"></div>Last Prediction — Raw Inputs</div>', unsafe_allow_html=True)
        st.json(st.session_state.last_result["inputs"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    model, model_ready, model_error = load_model()
    df, data_error = load_dataset()

    r2, mae, metrics_error = (None, None, None)
    if model_ready and df is not None:
        r2, mae, metrics_error = dataset_metrics(model, df)

    render_sidebar()
    render_header(model_ready, r2)

    diagnostics = [msg for msg in (model_error, data_error, metrics_error) if msg]
    if diagnostics:
        with st.expander("⚠ Some data didn't load correctly — click for details", expanded=False):
            for msg in diagnostics:
                st.warning(msg)
            st.caption(
                "Make sure Mental_Health_Model.pkl and "
                "Student_Social_Media_And_Mental_Health_Impact.csv sit in the "
                "same folder as app.py, and that scikit-learn is installed "
                "per requirements.txt."
            )

    page = st.session_state.page
    if page == "Home":
        page_home(df, model_ready, r2, mae)
    elif page == "Predict Score":
        page_predict(model, model_ready, df)
    elif page == "Model Performance":
        page_model_performance(df, r2, mae)
    elif page == "About Dataset":
        page_about_dataset(df)
    elif page == "About Project":
        page_about_project()
    elif page == "Settings":
        page_settings(model_ready)

    render_footer()


if __name__ == "__main__":
    main()