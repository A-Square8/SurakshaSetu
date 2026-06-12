from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class GradeOutput(BaseModel):
    relevant: bool = Field(description="Whether the document is relevant to the query")


def grading_node(state):
    q = state["rewritten_query"]
    docs = state["retrieved_docs"]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    structured_llm = llm.with_structured_output(GradeOutput)

    graded = []
    for doc in docs:
        prompt = f"Is this document relevant to the question: '{q}'?\n\nDocument: {doc.page_content[:500]}"
        res = structured_llm.invoke(prompt)
        if res.relevant:
            graded.append(doc)

    return {"graded_docs": graded, "sufficient": len(graded) > 0}
