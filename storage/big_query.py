from docs_process.parse_chunk_pdf import extract_text_from_file, chunk_text, embed_chunks
import json
pdf_pages = extract_text_from_file("C:\\Users\\Nandini\\Downloads\\Certificates\\imocha.pdf")
chunks = chunk_text(pdf_pages)
embeddings = embed_chunks(chunks)

# for p in embeddings:
    # print(f"Page {p['chunk_text']}: {p['embedding'][:100]}...\n")
    
from google.cloud import bigquery
import uuid
from datetime import datetime

PROJECT_ID = "agenticai-477010"
DATASET = "rag_pdf"
TABLE = "pdf_chunks"

client = bigquery.Client(project=PROJECT_ID)

TABLE_ID = f"{PROJECT_ID}.{DATASET}.{TABLE}"


def insert_embeddings(embedded_chunks, document_id):
    with open("temp_embeddings.json", "w") as f:
        for chunk in embedded_chunks:
            json.dump({
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "page_number": chunk["page_number"],
                "section": "NA",
                "content": chunk["chunk_text"],
                "embedding": chunk["embedding"].tolist() if hasattr(chunk["embedding"], "tolist") else chunk["embedding"],
                "created_at": datetime.now().isoformat()
            }, f)
            f.write("\n")

    # Load via batch job as bigquery insert is not allowed in free tier
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.autodetect = True

    with open("temp_embeddings.json", "rb") as source_file:
        job = client.load_table_from_file(source_file, TABLE_ID, job_config=job_config)

    job.result() 

# insert_embeddings(embedded_chunks=embeddings, document_id="C:\\Users\\Nandini\\Downloads\\Certificates\\imocha.pdf")