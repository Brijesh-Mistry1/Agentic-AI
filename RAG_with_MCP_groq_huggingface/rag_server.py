import os
from pypdf import PdfReader
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Groq-Chroma-Server")

class HuggingFaceTransformersEmbeddingFunction(EmbeddingFunction):
    """
    A custom embedding function that uses local Hugging Face 'transformers'
    to generate embeddings for semantic search.
    
    This replaces default ONNX MiniLM, running freely and entirely offline on your CPU/GPU.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def __call__(self, input: Documents) -> Embeddings:
        import torch

        inputs = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Generate model output without updating gradients
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply mean pooling to compute a single 384-dimension vector for each input text block
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        # Multiply token embeddings by the attention mask to zero out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Divide by sum of mask to get the mean
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings.cpu().tolist()


# Initialize ChromaDB persistent client and specify the custom Hugging Face embedding function
DB_DIR = "./chroma_data"
chroma_client = chromadb.PersistentClient(path=DB_DIR)
embedding_function = HuggingFaceTransformersEmbeddingFunction()

try:
    # If the collection exists, get it. Otherwise, create it.
    collection = chroma_client.get_or_create_collection(
        name="pdf_knowledge_base",
        embedding_function=embedding_function
    )
except ValueError as e:
    # If the database was created with a different embedding function previously, 
    # we delete and recreate it to avoid dimensional/metadata conflict.
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
    Performs a semantic search query on the indexed documents in the ChromaDB vector database
    and returns the top_k matching text blocks.
    
    Args:
        query (str): The search query in plain English.
        top_k (int): Number of most relevant documents to retrieve.
    """
    try:
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