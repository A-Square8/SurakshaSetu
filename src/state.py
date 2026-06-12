from typing import TypedDict, List, Any, Optional


class GraphState(TypedDict):
    question: str
    rewritten_query: str
    category: str
    retrieved_docs: List[Any]
    graded_docs: List[Any]
    sufficient: bool
    retry_count: int
    answer: str
    sources: List[str]
    hallucination_score: float
    chat_history: List[dict]
    session_id: Optional[str]
    web_search_used: bool
