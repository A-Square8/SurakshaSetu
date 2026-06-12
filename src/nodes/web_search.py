import os
from tavily import TavilyClient
from langchain.schema import Document


def web_search_node(state):
    q = state.get("rewritten_query", state["question"])
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return {"retrieved_docs": [], "web_search_used": True}

    client = TavilyClient(api_key=api_key)
    results = client.search(query=f"Indian law: {q}", max_results=5)

    docs = []
    for r in results.get("results", []):
        doc = Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "web"), "title": r.get("title", "")}
        )
        docs.append(doc)

    return {"retrieved_docs": docs, "web_search_used": True}
