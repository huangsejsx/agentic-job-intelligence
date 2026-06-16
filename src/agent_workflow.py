import json
from pathlib import Path
from typing import TypedDict, Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from tools import parse_jd_tool, parse_resume_tool, job_decision_tool

class JobAgentState(TypedDict, total=False):
    jd_path: str
    resume_path: str
    output_path: str
    jd_info: Dict[str, Any]
    resume_info: Dict[str, Any]
    decision: Dict[str, Any]
    status: str
    error: Optional[str]

def parse_jd_node(state: JobAgentState) -> JobAgentState:
    jd_info = parse_jd_tool(state["jd_path"])
    return {
        "jd_info": jd_info,
        "status": "JD parsed"
    }

def parse_resume_node(state: JobAgentState) -> JobAgentState:
    resume_info = parse_resume_tool(state["resume_path"])
    return {
        "resume_info": resume_info,
        "status": "Resume parsed"
    }

def decision_node(state: JobAgentState) -> JobAgentState:
    decision = job_decision_tool(state["jd_path"], state["resume_path"])
    return {
        "decision": decision,
        "status": "Decision completed"
    }

def save_result_node(state: JobAgentState) -> JobAgentState:
    result = {
        "jd_path": state["jd_path"],
        "resume_path": state["resume_path"],
        "jd_info": state["jd_info"],
        "resume_info": state["resume_info"],
        "decision": state["decision"]
    }
    output_path = Path(state["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return {
        "status": f"Saved result to {state['output_path']}"
    }

def build_job_agent_workflow():
    graph = StateGraph(JobAgentState)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("make_decision", decision_node)
    graph.add_node("save_result", save_result_node)
    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "parse_resume")
    graph.add_edge("parse_resume", "make_decision")
    graph.add_edge("make_decision", "save_result")
    graph.add_edge("save_result", END)
    return graph.compile()

def run_workflow(jd_path: str, resume_path: str, output_path: str) -> JobAgentState:
    app = build_job_agent_workflow()
    initial_state = {
        "jd_path": jd_path,
        "resume_path": resume_path,
        "output_path": output_path
    }
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    final_state = run_workflow(
        jd_path="data/job_descriptions/sample_jd_1.txt",
        resume_path="data/resumes/my_resume.txt",
        output_path="results/langgraph_sample_jd_1_decision.json"
    )
    print("Workflow finished.")
    print(f"Status: {final_state['status']}")
    print(f"Decision: {final_state['decision']['decision']}")
    print(f"Skill match score: {final_state['decision']['skill_match_score']}")