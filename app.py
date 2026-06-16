import sys
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from llm_jd_extractor import extract_jd_structured
from parse_resume import parse_resume
from semantic_decision_engine import make_semantic_decision
from parse_jd import parse_jd

st.set_page_config(
    page_title="Agentic Job Intelligence",
    page_icon="🧭",
    layout="wide"
)

st.title("Agentic Job Intelligence")
st.write("基于 structured JD extraction、semantic matching 和 calibrated decision rule 的岗位智能分析 Demo。")

default_jd = """About the job

Business Intelligence Intern - Marketplace Analytics

Who We Are

DemoShop is a fast-growing e-commerce platform connecting buyers, sellers, and creators across multiple regions.

Responsibilities

- Analyze user behavior, conversion funnels, retention, GMV, and seller performance.
- Build dashboards and automated reports to monitor business metrics.
- Work with product managers to design and evaluate A/B tests.
- Use SQL and Python to clean, transform, and analyze large-scale marketplace data.
- Support search, ranking, and recommendation-related analysis.
- Communicate insights clearly to technical and non-technical stakeholders.

Minimum Qualifications

- Currently pursuing a Bachelor's or Master's degree in Statistics, Mathematics, Computer Science, Data Science, Economics, or a related quantitative field.
- Strong SQL and Python skills.
- Familiarity with statistics, data analysis, data visualization, and experimental design.

Preferred Qualifications

- Experience with pandas, scikit-learn, Tableau, Power BI, or similar analytics tools.
- Interest in e-commerce, marketplace platforms, search, ranking, recommendation systems, or AI products.

Location

London, United Kingdom
"""

default_resume = """Alex Chen

Education
MSc Statistics, Example University
BSc Mathematics and Statistics

Skills
Python, SQL, Pandas, NumPy, Scikit-learn, Machine Learning, Deep Learning, Statistics, Data Analysis, A/B Testing, Recommender Systems, Ranking, RAG, LangChain, Streamlit

Projects
MiniOneRec-style Generative Recommendation Reproduction
Built a MovieLens-based recommendation system with implicit feedback preprocessing, temporal leave-one-out split, Popularity baseline, ItemCF baseline, Semantic ID construction, SID-Markov baseline, and Transformer-based sequential recommendation model.
Evaluated models using Recall@K, NDCG@K, coverage, head-tail distribution, and user/item group analysis.

Agentic Job Intelligence System
Built an AI Agent system using RAG, tool calling, structured extraction, hard-filter detection, skill matching, and explainable scoring.

Experience
Data Analysis Internship
Worked on data cleaning, business analysis, SQL queries, reporting, and performance tracking.
"""

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area("Job Description", value=default_jd, height=420)

with col2:
    resume_text = st.text_area("Resume", value=default_resume, height=420)

def run_analysis(jd_text: str, resume_text: str):
    structured_jd = extract_jd_structured(jd_text)
    resume_info = parse_resume(resume_text)
    rule_jd = parse_jd(jd_text)
    decision = make_semantic_decision(rule_jd, resume_info)
    trace = [
        "structured_jd_extraction",
        "parse_resume",
        "make_semantic_decision",
        "save_result"
    ]
    return structured_jd, resume_info, decision, trace

if st.button("Analyze Job Fit"):
    if not jd_text.strip() or not resume_text.strip():
        st.error("Please provide both JD text and resume text.")
    else:
        with st.spinner("Analyzing JD and resume..."):
            structured_jd, resume_info, decision, trace = run_analysis(jd_text, resume_text)

        st.subheader("Final Decision")

        decision_label = decision.decision
        if decision_label == "Apply":
            st.success(f"Decision: {decision_label}")
        elif decision_label == "Maybe":
            st.warning(f"Decision: {decision_label}")
        else:
            st.error(f"Decision: {decision_label}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Final Score", round(decision.final_score, 4))
        metric_col2.metric("Keyword Score", round(decision.keyword_score, 4))
        metric_col3.metric("Semantic Score", round(decision.semantic_score, 4))

        st.write(decision.explanation)

        st.subheader("Structured JD Extraction")
        jd_col1, jd_col2, jd_col3 = st.columns(3)
        jd_col1.write(f"**Title:** {structured_jd.title}")
        jd_col2.write(f"**Company:** {structured_jd.company}")
        jd_col3.write(f"**Location:** {structured_jd.location}")

        st.write("**Required Skills:**")
        st.write(", ".join(structured_jd.required_skills) if structured_jd.required_skills else "None")

        st.write("**Responsibilities:**")
        if structured_jd.responsibilities:
            for item in structured_jd.responsibilities:
                st.write(f"- {item}")
        else:
            st.write("None")

        st.subheader("Resume Parsing")
        st.write("**Skills extracted from resume:**")
        st.write(", ".join(resume_info.skills) if resume_info.skills else "None")

        st.subheader("Skill Matching")
        match_col1, match_col2 = st.columns(2)

        with match_col1:
            st.write("**Final Matched Skills**")
            if decision.final_matched_skills:
                st.write(", ".join(decision.final_matched_skills))
            else:
                st.write("None")

        with match_col2:
            st.write("**Final Missing Skills**")
            if decision.final_missing_skills:
                st.write(", ".join(decision.final_missing_skills))
            else:
                st.write("None")

        st.subheader("Semantic Evidence")
        evidence_rows = []
        for item in decision.semantic_evidence:
            evidence_rows.append({
                "JD Skill": item["jd_skill"],
                "Best Resume Evidence": item["best_resume_evidence"],
                "Similarity": item["similarity"],
                "Semantic Match": item["is_semantic_match"]
            })
        if evidence_rows:
            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)
        else:
            st.write("No semantic evidence available.")

        st.subheader("Workflow Trace")
        st.write(" -> ".join(trace))

        st.subheader("Notes")
        st.info(
            "This demo uses structured JD extraction, sentence-transformer semantic matching, "
            "and a calibrated decision threshold. Current thresholds are: "
            "Apply if score >= 0.85, Maybe if score >= 0.45, otherwise Pass. "
            "Hard filters such as PhD or many years of experience directly lead to Pass."
        )
