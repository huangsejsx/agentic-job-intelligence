from textwrap import dedent

from decision_engine import make_decision
from match_skills import (
    calculate_skill_match,
    calculate_weighted_skill_match,
    combine_required_preferred_scores
)
from parse_jd import parse_jd
from parse_resume import parse_resume
from semantic_decision_engine import choose_semantic_decision


def test_parse_jd_extracts_skills_and_hard_filters():
    jd_text = dedent("""
    Machine Learning Scientist
    Company: ExampleAI

    We are looking for a candidate with Python, SQL, statistics, and machine learning.
    Candidates must have a PhD and 5+ years of experience.
    """)

    jd_info = parse_jd(jd_text)

    assert jd_info.title == "Machine Learning Scientist"
    assert jd_info.company == "ExampleAI"
    assert {"python", "sql", "statistics", "machine learning"}.issubset(jd_info.required_skills)
    assert jd_info.experience_years == 5
    assert jd_info.requires_phd is True
    assert jd_info.requires_many_years is True
    assert jd_info.hard_filter_reason == "Requires PhD or doctoral-level qualification."


def test_parse_jd_separates_preferred_skills():
    jd_text = dedent("""
    Data Analyst Intern
    Company: ExampleCommerce

    Requirements:
    - Strong Python and SQL skills.
    - Familiarity with statistics and data analysis.

    Preferred Qualifications:
    - Experience with Tableau or Power BI is preferred.
    - Interest in recommendation systems is a plus.
    """)

    jd_info = parse_jd(jd_text)

    assert {"python", "sql", "statistics", "data analysis"}.issubset(jd_info.required_skills)
    assert {"tableau", "power bi", "recommendation"}.issubset(jd_info.preferred_skills)
    assert "tableau" not in jd_info.required_skills


def test_parse_resume_extracts_sections_and_skills():
    resume_text = dedent("""
    Alex Chen

    Education:
    MSc Statistics, Example University

    Skills:
    Python, SQL, Machine Learning, Streamlit

    Projects:
    Built a dashboard and machine learning analysis tool.

    Experience:
    Data Analysis Internship with SQL reporting.
    """)

    resume_info = parse_resume(resume_text)

    assert resume_info.name == "Alex Chen"
    assert resume_info.education == ["MSc Statistics, Example University"]
    assert "python" in resume_info.skills
    assert "sql" in resume_info.skills
    assert "machine learning" in resume_info.skills
    assert "streamlit" in resume_info.skills
    assert resume_info.projects == ["Built a dashboard and machine learning analysis tool."]
    assert resume_info.experiences == ["Data Analysis Internship with SQL reporting."]


def test_calculate_skill_match_scores_overlap():
    result = calculate_skill_match(
        jd_skills=["python", "sql", "tableau"],
        resume_skills=["python", "sql"]
    )

    assert result.matched_skills == ["python", "sql"]
    assert result.missing_skills == ["tableau"]
    assert result.skill_match_score == 0.6667


def test_calculate_weighted_skill_match_scores_required_and_preferred():
    result = calculate_weighted_skill_match(
        required_skills=["python", "sql"],
        preferred_skills=["tableau", "power bi"],
        resume_skills=["python", "sql", "tableau"]
    )

    assert result.required_result.skill_match_score == 1.0
    assert result.preferred_result.skill_match_score == 0.5
    assert result.weighted_score == 0.9


def test_combine_required_preferred_scores_handles_preferred_only():
    score = combine_required_preferred_scores(
        required_score=0.0,
        preferred_score=1.0,
        has_required_skills=False,
        has_preferred_skills=True
    )

    assert score == 1.0


def test_make_decision_respects_hard_filter():
    jd_info = parse_jd(dedent("""
    Machine Learning Scientist
    Company: ExampleAI
    Required: Python, SQL, machine learning, PhD, 5+ years of experience.
    """))
    resume_info = parse_resume(dedent("""
    Alex Chen

    Skills:
    Python, SQL, Machine Learning
    """))

    decision = make_decision(jd_info, resume_info)

    assert decision.decision == "Pass"
    assert decision.hard_filter_triggered is True
    assert decision.final_score == 0.0


def test_make_decision_apply_for_high_keyword_overlap():
    jd_info = parse_jd(dedent("""
    Data Analyst Intern
    Company: ExampleCommerce
    Required: Python, SQL, statistics, data analysis.
    """))
    resume_info = parse_resume(dedent("""
    Alex Chen

    Skills:
    Python, SQL, Statistics, Data Analysis
    """))

    decision = make_decision(jd_info, resume_info)

    assert decision.decision == "Apply"
    assert decision.hard_filter_triggered is False
    assert decision.final_score == 1.0


def test_make_decision_exposes_weighted_score_breakdown():
    jd_info = parse_jd(dedent("""
    Data Analyst Intern
    Company: ExampleCommerce

    Requirements:
    - Python and SQL.

    Preferred Qualifications:
    - Tableau.
    - Power BI.
    """))
    resume_info = parse_resume(dedent("""
    Alex Chen

    Skills:
    Python, SQL
    """))

    decision = make_decision(jd_info, resume_info)

    assert decision.final_score == 0.8
    assert decision.score_breakdown["required_skill_score"] == 1.0
    assert decision.score_breakdown["preferred_skill_score"] == 0.0
    assert decision.score_breakdown["required_weight"] == 0.8
    assert decision.score_breakdown["preferred_weight"] == 0.2
    assert decision.score_breakdown["matched_required_skill_count"] == 2
    assert decision.score_breakdown["missing_preferred_skill_count"] == 2


def test_semantic_decision_does_not_block_apply_on_missing_preferred_skills():
    decision, explanation = choose_semantic_decision(
        weighted_score=0.8,
        required_score=1.0
    )

    assert decision == "Apply"
    assert "preferred skills do not block" in explanation


def test_semantic_decision_requires_strong_required_score_for_apply():
    decision, explanation = choose_semantic_decision(
        weighted_score=0.8857,
        required_score=0.8571
    )

    assert decision == "Maybe"
    assert "required skill score is not high enough" in explanation
