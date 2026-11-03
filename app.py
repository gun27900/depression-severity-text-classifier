from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import (
    analyze_text,
    display_label,
    explain_prediction,
    load_model,
    thai_label,
)
from src.project_data import (
    RESULTS_DIR,
    project_metrics,
    read_csv,
    read_model_metadata,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "deployment_text_classifier.joblib"

LABEL_ORDER = ["Minimal", "Mild", "Moderate", "Severe"]
LABEL_COLORS = {
    "Minimal": "#1E9A6C",
    "Mild": "#4F7CE8",
    "Moderate": "#D99016",
    "Severe": "#D84A55",
}

EXAMPLES = {
    "เลือกตัวอย่าง": "",
    "Neutral": "Today was a normal day. I went to class, finished my work, and talked with my friends.",
    "Mild language": "I have been feeling a little sad lately, but I can still do my daily activities.",
    "Daily-life impact": "For the past few weeks, I have lost interest in things I used to enjoy and it is becoming difficult to focus on my work.",
    "High-risk language": "I feel completely hopeless and I keep thinking that life is not worth living.",
}


st.set_page_config(
    page_title="Depression Severity Classifier",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink:#12233D;
        --muted:#66758B;
        --soft:#8795A8;
        --line:#DCE6F2;
        --panel:#FFFFFF;
        --canvas:#F5F8FC;
        --brand:#2F6FED;
        --brand-dark:#174AAB;
        --brand-soft:#EDF4FF;
        --warning:#FFF7E6;
        --warning-line:#F1D18A;
    }

    .stApp { background: var(--canvas); }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#10213A 0%,#0D1B31 100%);
        border-right: 1px solid rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"] * { color:#EEF4FF; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#8EA2BF !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius:10px;
        padding:.58rem .75rem;
        transition:.15s ease;
        border:1px solid transparent;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background:rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background:rgba(71,124,240,.18);
        border-color:rgba(110,160,255,.18);
        box-shadow:inset 3px 0 0 #6EA0FF;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        display:none;
    }

    .brand-title { font-weight:850; font-size:1.08rem; margin:.15rem 0 .1rem; }
    .brand-sub { color:#8EA2BF; font-size:.78rem; line-height:1.55; margin-bottom:1.1rem; }
    .side-label { color:#7087A6; font-size:.68rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; margin-top:.85rem; }
    .side-value { color:#EAF1FC; font-size:.82rem; line-height:1.45; }

    .hero { padding:1rem 0 .75rem; margin-bottom:.35rem; }
    .eyebrow { color:var(--brand); font-weight:850; font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.45rem; }
    .hero h1 { color:var(--ink); font-size:3rem; margin:.15rem 0 .65rem; line-height:1.05; letter-spacing:-.04em; }
    .hero p { color:var(--muted); font-size:1rem; max-width:820px; line-height:1.72; margin-bottom:.75rem; }
    .badge { display:inline-block; background:var(--brand-soft); color:var(--brand-dark); border:1px solid #D2E1FF; border-radius:999px; padding:.34rem .62rem; margin:.2rem .28rem .2rem 0; font-size:.78rem; font-weight:700; }

    .notice {
        background:var(--warning);
        border:1px solid var(--warning-line);
        color:#715411;
        border-radius:14px;
        padding:.8rem 1rem;
        font-size:.9rem;
        line-height:1.6;
        margin:.55rem 0 1.25rem;
    }

    .input-shell {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:20px;
        padding:1.3rem 1.35rem .95rem;
        box-shadow:0 10px 30px rgba(26,51,84,.045);
        margin-bottom:1.2rem;
    }
    .section-kicker { color:var(--brand); font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.2rem; }
    .section-title { color:var(--ink); font-size:1.45rem; font-weight:850; margin:0 0 .25rem; }
    .section-note { color:var(--muted); font-size:.9rem; margin-bottom:.8rem; }

    .result-grid { display:grid; grid-template-columns:1.5fr .8fr .8fr; gap:.8rem; margin:.4rem 0 1rem; }
    .result-main, .result-mini {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:18px;
        padding:1.05rem 1.1rem;
        min-height:132px;
    }
    .result-main.uncertain { background:#FFF9ED; border-color:#F0D396; }
    .result-main.clearer { background:#F2F7FF; border-color:#CFE0FC; }
    .result-main.relative { background:#F5F8FF; border-color:#D8E4F8; }
    .status-pill { display:inline-block; border-radius:999px; padding:.28rem .56rem; font-size:.72rem; font-weight:800; letter-spacing:.02em; background:#E8EEF7; color:#45566F; margin-bottom:.55rem; }
    .result-main.uncertain .status-pill { background:#FBE8BA; color:#77570D; }
    .result-main.clearer .status-pill { background:#DDEAFF; color:#214F9C; }
    .result-main.relative .status-pill { background:#E5ECFA; color:#3A5684; }
    .result-label { color:var(--ink); font-size:1.85rem; font-weight:900; line-height:1.08; }
    .result-desc { color:var(--muted); font-size:.87rem; line-height:1.55; margin-top:.45rem; }
    .mini-label { color:var(--soft); font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
    .mini-value { color:var(--ink); font-size:1.75rem; font-weight:900; margin-top:.22rem; }
    .mini-note { color:var(--muted); font-size:.78rem; line-height:1.45; margin-top:.25rem; }

    .score-card { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:.9rem 1rem; min-height:112px; }
    .score-card .name { color:#64758A; font-weight:750; font-size:.86rem; }
    .score-card .value { color:var(--ink); font-size:1.6rem; font-weight:900; margin:.18rem 0 .28rem; }
    .score-card .sub { color:#8B98AA; font-size:.75rem; }
    .score-track { width:100%; height:7px; background:#EDF1F6; border-radius:999px; margin-top:.5rem; overflow:hidden; }
    .score-fill { height:100%; border-radius:999px; }

    .explain-shell { background:#FFFFFF; border:1px solid var(--line); border-radius:18px; padding:1rem 1.05rem; margin-top:.45rem; }
    .feature-row { display:grid; grid-template-columns:1fr 90px; gap:.8rem; align-items:center; padding:.55rem 0; border-bottom:1px solid #EDF1F5; }
    .feature-row:last-child { border-bottom:0; }
    .feature-name { color:var(--ink); font-weight:750; }
    .feature-score { text-align:right; color:#2E6A4F; font-variant-numeric:tabular-nums; font-weight:800; }

    .kpi { background:#FFFFFF; border:1px solid var(--line); border-radius:18px; padding:1.05rem; min-height:116px; }
    .kpi .label { color:var(--muted); font-size:.82rem; }
    .kpi .number { color:var(--ink); font-size:1.72rem; font-weight:900; margin:.22rem 0; }
    .kpi .note { color:#8794A6; font-size:.76rem; }


    .insight-card {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:18px;
        padding:1rem 1.05rem;
        height:100%;
    }
    .insight-title {
        color:var(--ink);
        font-size:1rem;
        font-weight:850;
        margin-bottom:.35rem;
    }
    .insight-text {
        color:#5C6C82;
        font-size:.90rem;
        line-height:1.68;
    }
    .metric-note {
        background:#EEF5FF;
        border:1px solid #D5E4FB;
        border-radius:14px;
        padding:.85rem 1rem;
        color:#315B91;
        font-size:.86rem;
        line-height:1.6;
        margin:.65rem 0 1rem;
    }


    .process-card {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:18px;
        padding:1.05rem 1rem;
        min-height:168px;
        height:100%;
    }
    .process-icon {
        width:34px;
        height:34px;
        border-radius:10px;
        background:#EAF2FF;
        color:var(--brand);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:.78rem;
        font-weight:900;
        margin-bottom:.8rem;
    }
    .process-title {
        color:var(--ink);
        font-weight:850;
        font-size:1rem;
        margin-bottom:.38rem;
    }
    .process-text {
        color:#5E6E84;
        font-size:.88rem;
        line-height:1.62;
    }
    .code-note {
        background:#0F213B;
        color:#DDE9FA;
        border-radius:16px;
        padding:1rem 1.1rem;
        font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size:.84rem;
        line-height:1.75;
        margin:.55rem 0 1rem;
    }
    .about-card {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:18px;
        padding:1rem 1.05rem;
        min-height:132px;
        height:100%;
    }
    .about-label {
        color:var(--brand);
        font-size:.70rem;
        font-weight:850;
        letter-spacing:.09em;
        text-transform:uppercase;
        margin-bottom:.35rem;
    }

    .step { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:1rem; height:100%; }
    .step-num { width:30px; height:30px; border-radius:9px; background:var(--brand); color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:850; margin-bottom:.7rem; }
    .step-title { color:var(--ink); font-weight:850; margin-bottom:.35rem; }
    .step-text { color:var(--muted); line-height:1.55; font-size:.88rem; }

    .footer { color:#8794A6; text-align:center; font-size:.78rem; padding:2rem 0 0; }
    .tiny-note { color:#7D8A9D; font-size:.8rem; line-height:1.55; }

    div[data-testid="stButton"] > button { border-radius:12px; min-height:2.9rem; font-weight:800; }
    textarea { border-radius:14px !important; }
    div[data-baseweb="select"] > div { border-radius:12px; }

    @media (max-width: 900px) {
        .hero h1 { font-size:2.35rem; }
        .result-grid { grid-template-columns:1fr; }
        .block-container { padding-left:1rem; padding-right:1rem; }
        [data-testid="stHorizontalBlock"] { gap:.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH)


@st.cache_data
def cached_metrics():
    return project_metrics()


@st.cache_data
def cached_csv(name: str):
    return read_csv(name)


def prediction_clarity(result: dict) -> tuple[str, str, str]:
    """Interpret model separation without presenting it as clinical confidence."""
    top = float(result["top_score"])
    margin = float(result["margin"])

    if result["uncertain"]:
        reasons = []
        if top < 0.50:
            reasons.append(f"คะแนนสูงสุดอยู่ที่ {top:.1%} ซึ่งยังต่ำกว่าเกณฑ์ 50%")
        if margin < 0.10:
            reasons.append(f"อันดับหนึ่งห่างจากอันดับสองเพียง {margin:.1%}")
        explanation = " และ ".join(reasons)
        return (
            "uncertain",
            "Uncertain",
            f"{explanation} จึงไม่ควรตีความว่าคลาสใดคลาสหนึ่งชัดเจน",
        )

    if top >= 0.65 and margin >= 0.15:
        return (
            "clearer",
            "Clearer prediction",
            f"คลาส {display_label(result['top_label'])} มีคะแนน {top:.1%} และนำอันดับสอง {margin:.1%} จึงแยกจากคลาสอื่นได้ค่อนข้างชัดภายในโมเดล",
        )

    return (
        "relative",
        "Relatively clear",
        f"คลาส {display_label(result['top_label'])} มีคะแนน {top:.1%} และนำอันดับสอง {margin:.1%} ผ่านเกณฑ์ขั้นต่ำของระบบ แต่ยังควรพิจารณาคะแนนคลาสอื่นร่วมด้วย",
    )


def apply_example() -> None:
    choice = st.session_state.get("example_choice", "เลือกตัวอย่าง")
    if choice in EXAMPLES and EXAMPLES[choice]:
        st.session_state["analysis_text"] = EXAMPLES[choice]


def clear_input() -> None:
    st.session_state["analysis_text"] = ""
    st.session_state["example_choice"] = "เลือกตัวอย่าง"


if "analysis_text" not in st.session_state:
    st.session_state["analysis_text"] = ""
if "example_choice" not in st.session_state:
    st.session_state["example_choice"] = "เลือกตัวอย่าง"


try:
    model = cached_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

metrics = cached_metrics()

with st.sidebar:
    st.markdown('<div class="brand-title">Severity Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">NLP • Machine Learning • Explainable AI</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Analyze", "Model Insights", "How It Works", "About Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-value">TF-IDF + Balanced Logistic Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Language</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-value">English</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Version</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-value">Portfolio v2.5</div>', unsafe_allow_html=True)


if page == "Analyze":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Interactive NLP application</div>
          <h1>Depression Severity<br>Text Classifier</h1>
          <p>Explore how an explainable machine-learning model classifies English text into four dataset labels, reports how clearly the classes are separated, and shows which TF-IDF features influenced the result.</p>
          <span class="badge">TF-IDF</span><span class="badge">Unigram + Bigram</span><span class="badge">Balanced Logistic Regression</span><span class="badge">Explainable ML</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="notice"><b>Educational use only.</b> ระบบนี้วิเคราะห์รูปแบบข้อความด้วย Machine Learning และไม่ใช่เครื่องมือวินิจฉัยหรือประเมินความเสี่ยงทางการแพทย์</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Analyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyze an English text</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Enter or select an example text to view class scores, prediction clarity and model explanation.</div>', unsafe_allow_html=True)

    st.selectbox(
        "Example",
        list(EXAMPLES.keys()),
        key="example_choice",
        on_change=apply_example,
    )

    text = st.text_area(
        "English text",
        key="analysis_text",
        height=210,
        placeholder="Type or paste English text here...",
    )
    st.caption(f"{len(text)} characters • English text only • Text is processed when you press Analyze")

    clear_col, analyze_col = st.columns([1, 5], gap="small")
    with clear_col:
        st.button("Clear", use_container_width=True, on_click=clear_input)
    with analyze_col:
        analyze = st.button("Analyze text", type="primary", use_container_width=True)

    if analyze:
        try:
            result = analyze_text(model, text)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            clarity_class, clarity_label, clarity_text = prediction_clarity(result)
            st.markdown("---")
            st.markdown('<div class="section-kicker">Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Analysis result</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="result-grid">
                  <div class="result-main {clarity_class}">
                    <span class="status-pill">{clarity_label}</span>
                    <div class="result-label">{display_label(result['top_label'])}</div>
                    <div class="result-desc">{clarity_text}</div>
                  </div>
                  <div class="result-mini">
                    <div class="mini-label">Highest model score</div>
                    <div class="mini-value">{result['top_score']:.1%}</div>
                    <div class="mini-note">คะแนนสูงสุดภายใน 4 คลาสของโมเดล</div>
                  </div>
                  <div class="result-mini">
                    <div class="mini-label">Top-two margin</div>
                    <div class="mini-value">{result['margin']:.1%}</div>
                    <div class="mini-note">ระยะห่างระหว่างอันดับหนึ่งและอันดับสอง</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Class scores")
            score_cols = st.columns(4)
            for col, label in zip(score_cols, LABEL_ORDER):
                score = float(result["scores"].get(label, 0.0))
                with col:
                    st.markdown(
                        f"""
                        <div class="score-card">
                          <div class="name">{label}</div>
                          <div class="value">{score:.1%}</div>
                          <div class="sub">{thai_label(label.lower())}</div>
                          <div class="score-track"><div class="score-fill" style="width:{score*100:.2f}%;background:{LABEL_COLORS[label]}"></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            detail_tab, explain_tab = st.tabs(["Score interpretation", "Why this result?"])

            with detail_tab:
                st.markdown("##### How to read this result")
                if result["uncertain"]:
                    st.write(
                        f"โมเดลให้คะแนน **{display_label(result['top_label'])}** สูงที่สุดที่ **{result['top_score']:.1%}** แต่ระบบยังจัดผลเป็น **Uncertain** เพราะ "
                        + " และ ".join(result["uncertainty_reasons"])
                        + "."
                    )
                else:
                    st.write(clarity_text + ".")

                st.write(
                    f"อันดับสองคือ **{display_label(result['second_label'])}** ที่ **{result['second_score']:.1%}** ดังนั้นระยะห่างระหว่างอันดับหนึ่งกับอันดับสองเท่ากับ **{result['margin']:.1%}**."
                )
                st.info(
                    "Model score ใช้เปรียบเทียบคลาสภายในโมเดลเท่านั้น ไม่ใช่โอกาสป่วยหรือระดับอาการของบุคคล และเกณฑ์ Uncertain เป็นกฎของระบบ Demo ไม่ใช่ clinical threshold"
                )

            with explain_tab:
                explanation = explain_prediction(model, text, result["top_label"], top_n=7)
                support = explanation["supporting"]
                oppose = explanation["opposing"]

                st.markdown(f"##### Features supporting {display_label(result['top_label'])}")
                st.write(
                    "ส่วนนี้แสดงคำหรือคำคู่ที่มีอยู่ในข้อความและมีส่วนเพิ่ม linear score ของคลาสที่ได้คะแนนสูงสุด"
                )

                if not support:
                    st.info("ไม่พบ feature ฝั่งสนับสนุนที่อยู่ใน vocabulary ของโมเดล")
                else:
                    rows = "".join(
                        f'<div class="feature-row"><div class="feature-name">{item["feature"]}</div><div class="feature-score">+{float(item["contribution"]):.4f}</div></div>'
                        for item in support
                    )
                    st.markdown(f'<div class="explain-shell">{rows}</div>', unsafe_allow_html=True)

                    top_features = ", ".join(f'“{item["feature"]}”' for item in support[:3])
                    st.caption(
                        f"ตัวอย่าง feature ที่มีอิทธิพลมากในข้อความนี้: {top_features}. Contribution = TF-IDF value × Logistic Regression coefficient."
                    )

                with st.expander(f"Features reducing the {display_label(result['top_label'])} score"):
                    if not oppose:
                        st.caption("ไม่พบ feature ที่ลดคะแนนของคลาสนี้")
                    else:
                        negative_rows = "".join(
                            f'<div class="feature-row"><div class="feature-name">{item["feature"]}</div><div class="feature-score" style="color:#8A4D55;">{float(item["contribution"]):.4f}</div></div>'
                            for item in oppose
                        )
                        st.markdown(f'<div class="explain-shell">{negative_rows}</div>', unsafe_allow_html=True)
                        st.caption(
                            "ค่าติดลบหมายถึง feature นั้นลด linear score ของคลาสนี้ในโมเดล ไม่ได้หมายความว่าคำนั้นมีความหมายเชิงบวกหรือเชิงลบทางการแพทย์"
                        )

                st.warning(
                    "Feature contribution = TF-IDF value × Logistic Regression coefficient และใช้อธิบายการคำนวณของโมเดลเท่านั้น ไม่ใช่สาเหตุ อาการ หรือหลักฐานทางการแพทย์"
                )

            st.markdown("##### Try another text")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.button("Clear and analyze again", use_container_width=True, on_click=clear_input)
            with c2:
                st.caption("เปลี่ยนข้อความด้านบนแล้วกด Analyze text อีกครั้งได้ทันที")


elif page == "Model Insights":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Evaluation & evidence</div>
          <h1>Model Insights</h1>
          <p>Explore the evidence behind the deployed model — dataset balance, model selection, cross-validation and final holdout performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    items = [
        (k1, "Clean records", f"{metrics['training_records']:,}", "หลังทำความสะอาดข้อมูล"),
        (k2, "5-Fold CV Macro F1", f"{metrics['cv_macro_f1']:.2%}", "ใช้เป็นเกณฑ์หลักในการเลือกโมเดล"),
        (k3, "Holdout Accuracy", f"{metrics['holdout_accuracy']:.2%}", "ประเมินบนข้อมูล 707 รายการ"),
        (k4, "Holdout Macro F1", f"{metrics['holdout_macro_f1']:.2%}", "ให้น้ำหนักแต่ละคลาสเท่ากัน"),
    ]
    for col, label, value, note in items:
        with col:
            st.markdown(
                f'<div class="kpi"><div class="label">{label}</div><div class="number">{value}</div><div class="note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="metric-note">
            <b>Why Macro F1?</b> Dataset มีความไม่สมดุลสูง โดยคลาส Minimal มีสัดส่วนมากที่สุด
            ดังนั้น Accuracy เพียงค่าเดียวอาจทำให้โมเดลที่ทายคลาสส่วนใหญ่บ่อย ๆ ดูดีเกินจริง
            โครงงานจึงใช้ Macro F1 เป็นตัวชี้วัดหลักในการเปรียบเทียบโมเดล
        </div>
        """,
        unsafe_allow_html=True,
    )

    dist_col, eval_col = st.columns([1.05, .95], gap="large")

    with dist_col:
        st.markdown("### Dataset distribution")
        distribution = cached_csv("clean_label_summary.csv")
        if not distribution.empty:
            dist = distribution.copy()
            dist["label"] = dist["label"].map(lambda x: display_label(str(x)))
            dist["percentage"] = pd.to_numeric(dist["percentage"], errors="coerce")
            st.bar_chart(
                dist.set_index("label")[["count"]],
                use_container_width=True,
            )
            table_dist = dist[["label", "count", "percentage"]].copy()
            table_dist.columns = ["Class", "Records", "Share (%)"]
            st.dataframe(
                table_dist,
                hide_index=True,
                use_container_width=True,
            )
            minimal_share = float(
                dist.loc[dist["label"] == "Minimal", "percentage"].iloc[0]
            ) if (dist["label"] == "Minimal").any() else 0.0
            st.caption(
                f"Minimal คิดเป็นประมาณ {minimal_share:.2f}% ของข้อมูลหลังทำความสะอาด จึงเกิด Class Imbalance อย่างชัดเจน"
            )

    with eval_col:
        st.markdown("### Evaluation strategy")
        strategy = [
            ("01", "Development split", "ใช้ข้อมูลส่วน Development สำหรับเปรียบเทียบวิธีและทำ 5-Fold Stratified Cross-validation"),
            ("02", "Model selection", "เลือกโมเดลจากค่าเฉลี่ย Macro F1 ไม่ได้เลือกจาก Accuracy อย่างเดียว"),
            ("03", "Holdout test", "ประเมินโมเดลที่เลือกบนชุด Holdout เพื่อรายงานผลสุดท้ายของโครงงาน"),
        ]
        for num, title, desc in strategy:
            st.markdown(
                f"""
                <div class="insight-card" style="margin-bottom:.65rem;">
                    <div class="section-kicker">{num}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model comparison")
    comparison = cached_csv("model_comparison.csv")
    cv_results = cached_csv("cross_validation_results.csv")

    if not comparison.empty:
        table = comparison[["model", "accuracy", "f1_macro"]].copy()
        table["Accuracy (%)"] = (table["accuracy"] * 100).round(2)
        table["Holdout Macro F1 (%)"] = (table["f1_macro"] * 100).round(2)
        table = table[["model", "Accuracy (%)", "Holdout Macro F1 (%)"]]

        chart = comparison[["model", "f1_macro"]].copy()
        chart["Model"] = chart["model"].replace(
            {
                "BoW + Logistic Regression": "BoW + LR",
                "BoW + Balanced Logistic Regression": "BoW + Balanced LR",
                "BoW Bigram + Balanced Logistic Regression": "BoW Bigram + Balanced LR",
                "Dummy majority baseline": "Dummy majority",
                "TF-IDF Bigram + Balanced Logistic Regression": "TF-IDF Bigram + Balanced LR",
            }
        )
        chart["Macro F1"] = chart["f1_macro"]
        chart["Macro F1 (%)"] = (chart["f1_macro"] * 100).round(2)

        comparison_chart = (
            alt.Chart(chart)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                y=alt.Y(
                    "Model:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=230, labelFontSize=12),
                ),
                x=alt.X(
                    "Macro F1:Q",
                    title="Macro F1",
                    scale=alt.Scale(domain=[0, 0.52]),
                    axis=alt.Axis(format=".0%"),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Macro F1 (%):Q", format=".2f"),
                ],
            )
            .properties(height=245)
        )
        st.altair_chart(comparison_chart, use_container_width=True)

        with st.expander("View comparison table"):
            st.dataframe(table, hide_index=True, use_container_width=True)

        st.info(
            "Dummy majority baseline มี Accuracy สูงเพราะ Dataset มี Minimal เป็นคลาสส่วนใหญ่ "
            "แต่ Macro F1 ต่ำมาก จึงเป็นตัวอย่างว่าทำไม Accuracy อย่างเดียวไม่เพียงพอสำหรับงานนี้"
        )

    if not cv_results.empty:
        selected = cv_results[
            cv_results["model"].str.contains("TF-IDF Bigram", case=False, na=False)
        ]
        if not selected.empty:
            row = selected.iloc[0]
            st.caption(
                f"โมเดลที่เลือกมี 5-Fold CV Macro F1 เฉลี่ย {float(row['cv_f1_macro_mean']):.2%} "
                f"± {float(row['cv_f1_macro_std']):.2%} และ CV Accuracy เฉลี่ย {float(row['cv_accuracy_mean']):.2%}."
            )

    st.markdown("### Holdout confusion matrix")
    left_cm, right_cm = st.columns([1.2, .8], gap="large")

    with left_cm:
        confusion_path = RESULTS_DIR / "final_test_confusion_matrix_normalized.png"
        if confusion_path.exists():
            st.image(
                str(confusion_path),
                caption="Normalized confusion matrix on the holdout test set",
                use_container_width=True,
            )

    with right_cm:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">What the matrix tells us</div>
                <div class="insight-text">
                    Minimal เป็นคลาสที่โมเดลแยกได้ดีที่สุด ขณะที่ Mild ยังเป็นคลาสที่ยากที่สุด
                    และมีการสับสนกับ Minimal มากกว่าอีกหลายคลาส ซึ่งสะท้อนข้อจำกัดของข้อมูล
                    และการใช้ TF-IDF กับโมเดลเชิงเส้นสำหรับบริบทภาษาที่ซับซ้อน
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-card" style="margin-top:.7rem;">
                <div class="insight-title">Selected model</div>
                <div class="insight-text">
                    <b>TF-IDF (Unigram + Bigram)</b><br>
                    Balanced Logistic Regression<br><br>
                    เลือกเพราะให้ Macro F1 สูงที่สุดในขั้นเปรียบเทียบ และยังสามารถอธิบาย
                    feature contribution จาก coefficient ของโมเดลได้โดยตรง
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Behavioral sanity test"):
        behavior = cached_csv("behavioral_models_summary.csv")
        if not behavior.empty:
            original = behavior[behavior["model"] == "Original deployment model"]
            if not original.empty:
                row = original.iloc[0]
                b1, b2, b3 = st.columns(3)
                b1.metric("Handcrafted texts", f"{int(row['test_count'])}")
                b2.metric("Marked uncertain", f"{int(row['uncertain_count'])}/{int(row['test_count'])}")
                b3.metric("Uncertain rate", f"{float(row['uncertain_percentage']):.0%}")
                st.warning(
                    "Behavioral test เป็น Sanity Check ของพฤติกรรมระบบเท่านั้น ไม่ใช่ Accuracy หรือ Clinical Validation "
                    "เพราะข้อความเหล่านี้ไม่ได้มี Ground Truth จากผู้เชี่ยวชาญ"
                )


elif page == "How It Works":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">System design</div>
          <h1>How It Works</h1>
          <p>What happens after a user submits English text — from numerical features to class scores, uncertainty and explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Prediction pipeline")
    steps = [
        ("01", "Text input", "รับข้อความภาษาอังกฤษจากผู้ใช้ โดยระบบจะประมวลผลเมื่อกด Analyze เท่านั้น"),
        ("02", "TF-IDF features", "Vectorizer แปลงคำเดี่ยวและคำคู่ (Unigram + Bigram) ให้เป็นเวกเตอร์เชิงตัวเลข"),
        ("03", "Balanced classifier", "Balanced Logistic Regression คำนวณ decision score ของ Minimal, Mild, Moderate และ Severe"),
        ("04", "Class scores", "แปลงผลเป็นคะแนนสำหรับเปรียบเทียบทั้ง 4 คลาสภายในโมเดล"),
        ("05", "Clarity check", "ตรวจ Top score และ Top-two margin เพื่อหลีกเลี่ยงการแสดงผลแบบฟันธงเมื่อคลาสใกล้กัน"),
        ("06", "Explanation", "แสดง feature contribution จาก TF-IDF value × coefficient เพื่ออธิบายการตัดสินใจของโมเดล"),
    ]

    for start_idx in range(0, len(steps), 3):
        cols = st.columns(3, gap="medium")
        for col, (num, title, desc) in zip(cols, steps[start_idx:start_idx + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="process-card">
                        <div class="process-icon">{num}</div>
                        <div class="process-title">{title}</div>
                        <div class="process-text">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Why this model?")
    why1, why2, why3 = st.columns(3, gap="medium")

    with why1:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-label">TF-IDF</div>
                <div class="insight-title">Strong classical NLP baseline</div>
                <div class="insight-text">
                    เหมาะกับข้อความแบบ sparse และให้น้ำหนักคำตามความสำคัญในเอกสาร
                    โดยใช้ทั้ง Unigram และ Bigram เพื่อเก็บข้อมูลจากคำเดี่ยวและคำคู่
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with why2:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-label">Balanced LR</div>
                <div class="insight-title">Handles class imbalance</div>
                <div class="insight-text">
                    Dataset มี Minimal มากกว่าคลาสอื่น จึงใช้ class_weight="balanced"
                    เพื่อให้น้ำหนักคลาสส่วนน้อยมากขึ้นระหว่างการฝึกโดยไม่สร้างข้อมูลสังเคราะห์
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with why3:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-label">Explainability</div>
                <div class="insight-title">Coefficients can be inspected</div>
                <div class="insight-text">
                    Logistic Regression เป็นโมเดลเชิงเส้น จึงสามารถนำ coefficient
                    มาร่วมกับค่า TF-IDF เพื่ออธิบาย feature ที่เพิ่มหรือลดคะแนนของแต่ละคลาสได้
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Prediction clarity")
    st.markdown(
        """
        <div class="code-note">
        Uncertain&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; top score &lt; 50% &nbsp;OR&nbsp; top-two margin &lt; 10%<br>
        Relatively clear&nbsp;&nbsp; passes the minimum uncertainty rule<br>
        Clearer prediction&nbsp; top score ≥ 65% &nbsp;AND&nbsp; top-two margin ≥ 15%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "เกณฑ์ Prediction clarity เป็นกฎของ Demo เพื่ออธิบายว่าคะแนนของคลาสแยกออกจากกันชัดเพียงใด "
        "ไม่ได้เป็น Clinical threshold และไม่ได้ผ่านการ Calibration เพื่อใช้วินิจฉัยทางการแพทย์"
    )

    st.markdown("### Explainability calculation")
    st.latex(
        r"\mathrm{Feature\ Contribution}_{j,c}"
        r"=\mathrm{TFIDF}_{j}\times\beta_{j,c}"
    )
    st.write(
        "เมื่อ feature มี contribution เป็นบวก จะเพิ่ม linear score ของคลาสนั้น "
        "ส่วนค่าติดลบจะลด linear score ของคลาสนั้น การตีความนี้อธิบายกลไกของโมเดล "
        "ไม่ได้หมายความว่า feature นั้นเป็นอาการหรือสาเหตุทางการแพทย์"
    )

    st.markdown("### What the system does not do")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("**No diagnosis**")
        st.caption("ไม่ระบุว่าผู้เขียนข้อความมีหรือไม่มีโรคซึมเศร้า")
    with n2:
        st.markdown("**No clinical risk score**")
        st.caption("เปอร์เซ็นต์ที่แสดงเป็นคะแนนระหว่างคลาสของโมเดล ไม่ใช่ความเสี่ยงทางการแพทย์")
    with n3:
        st.markdown("**No hidden LLM**")
        st.caption("ผลหลักมาจาก TF-IDF + Logistic Regression ที่ฝึกไว้ ไม่ได้เรียก LLM มาตัดสินผลเบื้องหลัง")


else:
    metadata = read_model_metadata()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Project profile</div>
          <h1>About the Project</h1>
          <p>An end-to-end NLP portfolio project covering data preparation, model comparison, evaluation, explainability and interactive deployment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project overview")
    st.write(
        "**การใช้ปัญญาประดิษฐ์ในการจำแนกระดับความรุนแรงของภาวะซึมเศร้าจากข้อความ**  \n"
        "*The Use of Artificial Intelligence for Classifying Depression Severity Levels from Text*"
    )
    st.write(
        "ระบบรับข้อความภาษาอังกฤษและจำแนกตาม 4 labels ของ Reddit Depression Dataset ได้แก่ "
        "**Minimal, Mild, Moderate และ Severe** โดยเน้นการสาธิตกระบวนการ Machine Learning "
        "แบบครบวงจรและการอธิบายผลของโมเดล"
    )

    st.markdown("### What this project demonstrates")
    a1, a2, a3, a4 = st.columns(4, gap="small")
    cards = [
        (a1, "Data", "Preparation", "ตรวจข้อมูลซ้ำ วิเคราะห์ Class Imbalance และเตรียมข้อมูลสำหรับการทดลอง"),
        (a2, "ML", "Model selection", "สร้าง Baseline เปรียบเทียบหลายวิธี และเลือกโมเดลด้วย Macro F1"),
        (a3, "Eval", "Evaluation", "ใช้ Stratified 5-Fold CV, Holdout Test, Confusion Matrix และ Error Analysis"),
        (a4, "Deploy", "Explainable app", "นำโมเดลที่เลือกไปใช้กับ Streamlit พร้อม Uncertainty และ Feature Contribution"),
    ]
    for col, label, title, desc in cards:
        with col:
            st.markdown(
                f"""
                <div class="about-card">
                    <div class="about-label">{label}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Technical profile")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Tech stack")
        st.markdown(
            '<span class="badge">Python</span>'
            '<span class="badge">scikit-learn</span>'
            '<span class="badge">pandas</span>'
            '<span class="badge">Streamlit</span>'
            '<span class="badge">joblib</span>'
            '<span class="badge">Altair</span>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Model")
        st.write(f"**Method:** {metadata.get('method', metrics['method'])}")
        st.write(f"**Training records:** {int(metadata.get('training_records', metrics['training_records'])):,}")
        st.write(f"**Vocabulary size:** {int(metadata.get('vocabulary_size', metrics['vocabulary_size'])):,} features")
        st.write("**Language:** English")
        st.write("**Classes:** Minimal, Mild, Moderate, Severe")

    with right:
        st.markdown("#### Evaluation highlights")
        st.write(f"**5-Fold CV Macro F1:** {metrics['cv_macro_f1']:.2%}")
        st.write(f"**Holdout Accuracy:** {metrics['holdout_accuracy']:.2%}")
        st.write(f"**Holdout Macro F1:** {metrics['holdout_macro_f1']:.2%}")
        st.write("**Selection metric:** Macro F1")
        st.caption(
            "Macro F1 ถูกใช้เป็นตัวชี้วัดหลักเนื่องจาก Dataset มี Class Imbalance "
            "และต้องการให้น้ำหนักประสิทธิภาพของแต่ละคลาสอย่างเท่าเทียม"
        )

    st.markdown("### Known limitations")
    l1, l2 = st.columns(2, gap="large")

    with l1:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Data & model limitations</div>
                <div class="insight-text">
                    • Dataset มี Class Imbalance<br>
                    • Mild เป็นคลาสที่โมเดลจำแนกได้ยากที่สุด<br>
                    • TF-IDF ไม่เข้าใจบริบทเชิงลึก การประชด หรือผู้ถูกกล่าวถึงแบบมนุษย์<br>
                    • ข้อความที่แตกต่างจากรูปแบบ Reddit อาจให้ผลไม่แน่ชัด
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with l2:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Responsible-use boundary</div>
                <div class="insight-text">
                    ระบบนี้เป็น Educational / Portfolio prototype คะแนนของโมเดลใช้สำหรับ
                    เปรียบเทียบระหว่าง labels ภายในระบบเท่านั้น ไม่ใช่ Clinical probability,
                    diagnosis หรือเครื่องมือตรวจจับภาวะฉุกเฉิน
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Project status")
    st.success(
        "Current build: Portfolio v2.5 • Model evaluation completed • Explainability enabled • "
        "Interactive web application ready for deployment"
    )


st.markdown(
    '<div class="footer">Educational NLP portfolio project • Model output is not a medical diagnosis</div>',
    unsafe_allow_html=True,
)
