import json
from pathlib import Path
from typing import TypedDict, Any, Dict, Optional, List
from langgraph.graph import StateGraph, START, END
from tools import structured_jd_extraction_tool, parse_resume_tool, semantic_job_decision_tool

class JobAgentState(TypedDict, total=False):
    jd_path: str
    resume_path: str
    output_path: str
    jd_info: Dict[str, Any]
    resume_info: Dict[str, Any]
    decision: Dict[str, Any]
    status: str
    error: Optional[str]
    trace: List[str]

def append_trace(state: JobAgentState, node_name: str) -> List[str]:
    return state.get("trace", []) + [node_name]

def structured_jd_node(state: JobAgentState) -> JobAgentState:
    jd_info = structured_jd_extraction_tool(state["jd_path"])
    return {
        "jd_info": jd_info,
        "status": "Structured JD extracted",
        "trace": append_trace(state, "structured_jd_extraction")
    }

def parse_resume_node(state: JobAgentState) -> JobAgentState:
    resume_info = parse_resume_tool(state["resume_path"])
    return {
        "resume_info": resume_info,
        "status": "Resume parsed",
        "trace": append_trace(state, "parse_resume")
    }

def route_after_structured_jd(state: JobAgentState) -> str:
    jd_info = state["jd_info"]
    if jd_info["requires_phd"] or jd_info["requires_many_years"]:
        return "hard_filter_pass"
    return "parse_resume"

def hard_filter_pass_node(state: JobAgentState) -> JobAgentState:
    jd_info = state["jd_info"]
    reason = None
    if jd_info.get("hard_constraints"):
        reason = "; ".join(jd_info["hard_constraints"])
    if not reason:
        reason = "Hard filter triggered."
    decision = {
        "decision": "Pass",
        "final_score": 0.0,
        "hard_filter_triggered": True,
        "hard_filter_reason": reason,
        "keyword_score": 0.0,
        "semantic_score": 0.0,
        "keyword_matched_skills": [],
        "keyword_missing_skills": jd_info["required_skills"],
        "final_matched_skills": [],
        "final_missing_skills": jd_info["required_skills"],
        "semantic_evidence": [],
        "explanation": f"Pass because the role has a hard requirement: {reason}"
    }
    return {
        "decision": decision,
        "status": "Hard filter triggered",
        "trace": append_trace(state, "hard_filter_pass")
    }

def semantic_decision_node(state: JobAgentState) -> JobAgentState:
    decision = semantic_job_decision_tool(state["jd_path"], state["resume_path"])
    return {
        "decision": decision,
        "status": "Semantic decision completed",
        "trace": append_trace(state, "make_semantic_decision")
    }

def save_result_node(state: JobAgentState) -> JobAgentState:
    trace = append_trace(state, "save_result")
    result = {
        "jd_path": state["jd_path"],
        "resume_path": state["resume_path"],
        "jd_info": state["jd_info"],
        "resume_info": state.get("resume_info"),
        "decision": state["decision"],
        "trace": trace
    }
    output_path = Path(state["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return {
        "status": f"Saved result to {state['output_path']}",
        "trace": trace
    }

def build_structured_job_agent_workflow():
    graph = StateGraph(JobAgentState)
    graph.add_node("structured_jd_extraction", structured_jd_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("hard_filter_pass", hard_filter_pass_node)
    graph.add_node("make_semantic_decision", semantic_decision_node)
    graph.add_node("save_result", save_result_node)
    graph.add_edge(START, "structured_jd_extraction")
    graph.add_conditional_edges(
        "structured_jd_extraction",
        route_after_structured_jd,
        {
            "hard_filter_pass": "hard_filter_pass",
            "parse_resume": "parse_resume"
        }
    )
    graph.add_edge("hard_filter_pass", "save_result")
    graph.add_edge("parse_resume", "make_semantic_decision")
    graph.add_edge("make_semantic_decision", "save_result")
    graph.add_edge("save_result", END)
    return graph.compile()

def run_structured_workflow(jd_path: str, resume_path: str, output_path: str) -> JobAgentState:
    app = build_structured_job_agent_workflow()
    initial_state = {
        "jd_path": jd_path,
        "resume_path": resume_path,
        "output_path": output_path,
        "trace": []
    }
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    print("=== Structured Workflow: complex JD ===")
    state = run_structured_workflow(
        jd_path="data/job_descriptions/sample_jd_4_complex_format.txt",
        resume_path="data/resumes/my_resume.txt",
        output_path="results/langgraph_structured_sample_jd_4.json"
    )
    print(f"Status: {state['status']}")
    print(f"Title: {state['jd_info']['title']}")
    print(f"Company: {state['jd_info']['company']}")
    print(f"Location: {state['jd_info']['location']}")
    print(f"Decision: {state['decision']['decision']}")
    print(f"Keyword score: {state['decision']['keyword_score']}")
    print(f"Semantic score: {state['decision']['semantic_score']}")
    print(f"Trace: {' -> '.join(state['trace'])}")