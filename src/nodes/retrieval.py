from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def retrieval_node(state):
    q = state["rewritten_query"]
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory="data/chroma", embedding_function=emb)
    
    docs = db.similarity_search(q, k=5)
    return {"retrieved_docs": docs}
