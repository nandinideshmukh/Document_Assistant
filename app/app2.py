import streamlit as st
import asyncio
import os, sys
import tempfile,logging
import nest_asyncio
nest_asyncio.apply()   
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
sys.stdout = sys.stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from storage.big_query import insert_embeddings
from docs_process.parse_chunk_pdf import extract_text_from_file, chunk_text, embed_chunks

st.set_page_config(page_title="MCP LangGraph Chat", layout="wide")
st.title("🤖 MCP + LangGraph Document Assistant")

# launch mcp server
server_params = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(PROJECT_ROOT, "mcp_protocol", "mcp_server.py")],   
    env={
        **os.environ, 
        "PYTHONPATH": PROJECT_ROOT,  
    },)

st.sidebar.header("Document Management")
with st.sidebar:
    upload_file = st.file_uploader("Upload a PDF to BigQuery", type="pdf")
    if upload_file is not None:
        if st.button("upload Document"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(upload_file.getvalue())
                    tmp_path = tmp.name
                
                pages = extract_text_from_file(tmp_path)
                chunks = chunk_text(pages)
                embeddings = embed_chunks(chunks)
                insert_embeddings(embeddings, document_id=upload_file.name)
                
                st.success("Document indexed in BigQuery!")
                os.remove(tmp_path)
                
if "messages" not in st.session_state:
    st.session_state.messages = []

# calling the mcp tool
async def call_mcp_server(query):
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("rag_tool", arguments={"question": query})
                return result.content[0].text
    except asyncio.TimeoutError:
        return "MCP server timed out during initialization."
    except Exception as e:
        return f"Error connecting to MCP server: {str(e)}"   

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your indexed documents"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Please wait for few seconds ..."):
            try:
                response = asyncio.run(call_mcp_server(prompt))
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Failed to reach MCP Server: {e}")
                st.info("Check if mcp_protocol.mcp_server is running and PYTHONPATH is correct.")