from llm.llm import get_query_embedding,retrieve_relevant_chunks,generate_answer_using_llm,RAGState, StateGraph

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("query_embedding", get_query_embedding)
    graph.add_node("retrieve_chunks", retrieve_relevant_chunks)
    graph.add_node("generate_answer", generate_answer_using_llm)

    graph.set_entry_point("query_embedding")
    graph.add_edge("query_embedding", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "generate_answer")
    graph.set_finish_point("generate_answer")
    
    rag = graph.compile()
    return rag

# rag_app = build_rag_graph()
# result = rag_app.invoke({
#     "question": "Can the resume be applicable for internship with domain as backend?",
#     # "query_embedding": [],
#     # "context_chunks": [],
#     # "answer": ""
# })


# print(result["answer"])


