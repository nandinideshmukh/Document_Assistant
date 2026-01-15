from storage.big_query import insert_embeddings # for paid version only
from langgraph.graph import StateGraph,END
from typing import TypedDict, List
from google.cloud import bigquery
import os
from docs_process.parse_chunk_pdf import EMBEDDING_MODEL,client
from google import genai
import dotenv
dotenv.load_dotenv()

class RAGState(TypedDict):
    question: str
    query_embedding: List[float]
    context_chunks: List
    answer: str


PROJECT_ID = "agenticai-477010"
TABLE_ID = "agenticai-477010.rag_pdf.pdf_chunks"

bq_client = bigquery.Client(project=PROJECT_ID)

LLM_MODEL = "gemini-2.5-flash"

def get_query_embedding(state: RAGState):
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=state["question"],
            config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        embedding = response.embeddings[0].values
        return {"query_embedding": embedding}
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return {"query_embedding": []}

# big query support for vector similarity search and also dialect of sql
def retrieve_relevant_chunks(state: RAGState, top_k=3):
    # Use ML.DISTANCE and ensure 'content' is explicitly selected
    query = f"""
    SELECT content 
    FROM `{TABLE_ID}`
    WHERE embedding IS NOT NULL
    ORDER BY ML.DISTANCE(embedding, @query_embedding, 'COSINE') ASC
    LIMIT {top_k}
    """
    
    try:
        job = bq_client.query(query, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", state["query_embedding"])
            ]
        ))
        
        # result() ensures the query finishes before we iterate
        rows = job.result() 
        
        contexts = [row.get("content") for row in rows]
        
        if not contexts:
            print("Warning: No relevant chunks found in BigQuery.")
            
        return {"context_chunks": contexts}
        
    except Exception as e:
        print(f"Error during BigQuery retrieval: {e}")
        return {"context_chunks": []}

def generate_answer_using_llm(state: RAGState):
    # Add a safety check for context
    if not state.get("context_chunks"):
        return {"answer": "I couldn't find any relevant information in the documents to answer your question."}

    context = "\n\n".join(state["context_chunks"])
    prompt = f"""
    You are a document assistant.
    Answer ONLY using the context below.
    You cannot use any prior knowledge.
    But during answer can add only 20 words of your own knowledge if needed.
    Context:
    {context}

    Question:
    {state["question"]}
    """

    try:
        # Use the model name directly
        response = client.models.generate_content(
            model=LLM_MODEL, 
            contents=prompt
        )
        return {"answer": response.text}
    except Exception as e:
        print(f"Error in generation: {e}")
        return {"answer": "Error generating response."}

# for model in client.models.list():
    # print(model.name)