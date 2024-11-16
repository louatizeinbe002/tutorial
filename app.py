from flask import Flask

import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from chromadb import Client, Settings
from typing import List, Optional
from langchain.schema import Document
from script_name import query_chroma, load_pdfs, split_text, add_to_chroma, PDF_FILES
from langchain_huggingface import HuggingFaceEndpoint

model: Optional[HuggingFaceEndpoint] = None

app = Flask(__name__)

# Initialize the database
@app.route("/initialize", methods=["POST"])
async def initialize():
    pdf_text = load_pdfs(PDF_FILES)
    web_text = await load_web_pages(WEB_URLS)
    combined_text = pdf_text + web_text
    chunks = split_text(combined_text)
    add_to_chroma(chunks)
    return jsonify({"message": "Database initialized with PDFs and web data successfully!"})

# Query endpoint
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    result = query_chroma(query_text)
    return jsonify(result)


# Paths
CHROMA_PATH = "chroma"
PDF_FILES = ["/content/9135b.pdf", "/content/Chessbook - Irving Chernev - Logical Chess - Move by Move.pdf"]
WEB_URLS = [
         "https://www.chess.com/article/view/how-to-set-up-a-chessboard",
        "https://www.chess.com/terms/chess-pieces",
        "https://www.chess.com/terms/check-chess",
        "https://www.chess.com/terms/checkmate-chess",
        "https://www.chess.com/terms/draw-chess",
        "https://www.chess.com/terms/en-passant",
        "https://www.chess.com/terms/pawn-promotion",
        "https://www.chess.com/terms/castling-chess",
        "https://www.chess.com/terms/castling-chess",
]
# Initialize ChromaDB client
chroma_client = Client(Settings(persist_directory=CHROMA_PATH))

# Embedding function
def get_embedding_function():
    return OllamaEmbeddings(model="mxbai-embed-large")

# Load PDF files into text
def load_pdfs(file_paths):
    combined_text = ""
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            combined_text += doc.page_content + "\n"
    return combined_text

# Load text from web URLs
async def load_web_pages(urls):
    combined_text = ""
    for url in urls:
        loader = UnstructuredLoader(web_url=url)
        async for doc in loader.alazy_load():
            combined_text += doc.page_content + "\n"
    return combined_text

# Split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return splitter.split_text(text)

# Add chunks to ChromaDB
def add_to_chroma(chunks):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Add documents with unique IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"id": f"chunk_{i}"}
    db.add_documents(chunks)
    db.persist()

# Query ChromaDB with RAG
def query_chroma(query_text):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Prepare and run the prompt
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    
    {context}
    
    ---
    
    Question: {question}
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B",
        temperature=0.7,
        model_kwargs={
            "api_key": HUGGINGFACE_API_KEY,
            }
        )
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _ in results]

    return {"response": response_text, "sources": sources}

# Main function
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query text for RAG.", required=True)
    args = parser.parse_args()

    if not os.path.exists(CHROMA_PATH):
        print("Initializing database...")
        pdf_text = load_pdfs(PDF_FILES)
        web_text = await load_web_pages(WEB_URLS)
        combined_text = pdf_text + web_text
        chunks = split_text(combined_text)
        add_to_chroma(chunks)
        print("Database initialized!")

    print("Querying...")
    result = query_chroma(args.query)
    print(result)



if __name__ == "__main__":
    app.run(debug=True)

"""curl -X POST http://127.0.0.1:5000/initialize
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "How does the king move in chess?"}'
"""

