import os
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()


mcp = FastMCP("Groq-Chroma-Server")

DB_DIR = "./chroma_data"
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Use ChromaDB's default local ONNX MiniLM model for embeddings (runs locally, free and fast)
embedding_function = ONNXMiniLM_L6_V2()
try:
    collection = chroma_client.get_or_create_collection(
        name="pdf_knowledge_base",
        embedding_function=embedding_function
    )
except ValueError as e:
    if "Embedding function conflict" in str(e):
        chroma_client.delete_collection("pdf_knowledge_base")
        collection = chroma_client.get_or_create_collection(
            name="pdf_knowledge_base",
            embedding_function=embedding_function
        )
    else:
        raise e


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


@mcp.tool()
def ingest_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"Error: The target file '{file_path}' does not exist."
    
    try:
        reader = PdfReader(file_path)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
                
        if not extracted_text.strip():
            return "Error: Unable to extract readable strings from file layout."

        chunks = chunk_text(extracted_text)
        filename = os.path.basename(file_path)
        
        # Prepare payloads for ChromaDB bulk addition (ChromaDB computes embeddings locally)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        return f"Success: Parsed '{filename}'. Added {len(chunks)} text objects to ChromaDB store."
    
    except Exception as e:
        return f"Extraction pipeline exception: {str(e)}"


@mcp.tool()
def query_vector_db(query: str, top_k: int = 3) -> str:
    """
    Queries ChromaDB collections with semantic search queries 
    and returns relevant matching documents.
    """
    try:
        # ChromaDB automatically embeds the query text locally
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return "No matching context found inside the vector database"
            
        formatted_blocks = []
        for doc, meta in zip(documents, metadatas):
            formatted_blocks.append(f"[Source: {meta.get('source')}]\nContext: {doc}")
            
        return "\n\n---\n\n".join(formatted_blocks)
        
    except Exception as e:
        return f"Vector query operation failed: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")