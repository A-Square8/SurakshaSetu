from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class HallucinationOutput(BaseModel):
    grounded: bool = Field(description="Whether the answer is fully supported by the provided documents")
    reasoning: str = Field(description="Brief explanation of why the answer is or isn't grounded")


def hallucination_check_node(state):
    answer = state.get("answer", "")
    docs = state.get("graded_docs", [])

    if not docs or not answer:
        return {"hallucination_score": 0.0}

    ctx = "\n\n".join([d.page_content for d in docs])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    structured_llm = llm.with_structured_output(HallucinationOutput)

    prompt = (
        f"You are a fact-checker. Determine if the following answer is fully supported "
        f"by the provided source documents. The answer should not contain claims that "
        f"aren't backed by the documents.\n\n"
        f"Answer: {answer}\n\n"
        f"Source Documents:\n{ctx}"
    )
    res = structured_llm.invoke(prompt)

    return {"hallucination_score": 1.0 if res.grounded else 0.0}
