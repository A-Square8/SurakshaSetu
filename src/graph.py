from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes.query_analysis import query_analysis_node
from src.nodes.retrieval import retrieval_node
from src.nodes.grading import grading_node
from src.nodes.generation import generation_node
from src.nodes.hallucination import hallucination_check_node


def route_after_grading(state):
    if state.get("sufficient"):
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "retry"


def route_after_hallucination(state):
    if state.get("hallucination_score", 0.0) >= 0.5:
        return "end"
    if state.get("retry_count", 0) >= 2:
        return "end"
    return "regenerate"


def retry_node(state):
    count = state.get("retry_count", 0)
    return {"retry_count": count + 1}


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("query_analysis", query_analysis_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("grading", grading_node)
    g.add_node("retry", retry_node)
    g.add_node("generation", generation_node)
    g.add_node("hallucination_check", hallucination_check_node)

    g.set_entry_point("query_analysis")
    g.add_edge("query_analysis", "retrieval")
    g.add_edge("retrieval", "grading")

    g.add_conditional_edges("grading", route_after_grading, {
        "generate": "generation",
        "retry": "retry"
    })

    g.add_edge("retry", "query_analysis")
    g.add_edge("generation", "hallucination_check")

    g.add_conditional_edges("hallucination_check", route_after_hallucination, {
        "end": END,
        "regenerate": "generation"
    })

    return g.compile()


app = build_graph()
