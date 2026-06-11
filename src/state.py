from typing import TypedDict, List, Any

class GraphState(TypedDict):
    question: str
    rewritten_query: str
    category: str
    retrieved_docs: List[Any]
    graded_docs: List[Any]
    retry_count: int
    answer: str
    sources: List[str]
