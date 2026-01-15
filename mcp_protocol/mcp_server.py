import sys
import logging
from mcp.server.fastmcp import FastMCP
from rag.nodes import build_rag_graph

# Redirect logs to stderr so they don't corrupt the MCP pipe
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

rag_app = build_rag_graph()

mcp = FastMCP("BigQuery RAG Server")

@mcp.tool()
def rag_tool(question: str) -> str:
    """Answers document questions using BigQuery RAG."""
    logger.info(f"Received question: {question}")
    # Initialize state for the graph
    initial_state = {
        "question": question
        # "query_embedding": [],
        # "context_chunks": [],
        # "answer": ""
    }
    result = rag_app.invoke(initial_state)
    return result.get("answer", "No answer generated.")

if __name__ == "__main__":
    mcp.run()