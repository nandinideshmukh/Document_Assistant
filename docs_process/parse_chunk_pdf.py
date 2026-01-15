import dotenv
import pymupdf ,os # PyMuPDF
from llama_index.core.node_parser   import SentenceSplitter
from google.genai.types import EmbedContentConfig
import dotenv
from google import genai

def extract_text_from_file(pdf_file) -> str:
    """Extract text from an uploaded PDF file."""
    try:
        doc = pymupdf.open(pdf_file)
        pages = []

        text = ""
        for i,page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_number": i + 1,
                "text": text
            })
        # print("total text extracted:", len(text))        
        return pages
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
def chunk_text(text: str) -> list:
    """Chunk text into smaller pieces."""
    chunker = SentenceSplitter(chunk_size=100, chunk_overlap=50)
    all_chunks = []
    for page in text:
        page_text = page['text']
        page_chunks = chunker.split_text(page_text)
        for chunk in page_chunks:
            all_chunks.append({
                "page_number": page['page_number'],
                "chunk_text": chunk
            })
    # print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
    # redundant code below
    chunks = chunker.chunk(text)
    return chunks

dotenv.load_dotenv()
EMBEDDING_MODEL = "models/text-embedding-004"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> list:
    """
    Generate a 768-dimensional embedding for a text chunk using Gemini.
    """
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
           config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        embedding = response.embeddings[0].values
        return embedding
    
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []
    
def embed_chunks(chunks: list) -> list:
    """Embed each text chunk using Gemini embeddings."""
    embedded_chunks = []
    for chunk in chunks:
        embedding = get_embedding(chunk['chunk_text'])
        embedded_chunks.append({
            "page_number": chunk['page_number'],
            "chunk_text": chunk['chunk_text'],
            "embedding": embedding
        })
        
    # print(f"Total embedded chunks: {len(embedded_chunks)}")
    return embedded_chunks


