import objectbox
from flask import Flask, request, jsonify
import os
import argparse
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import pickle
from objectbox import *

app = Flask(__name__)

# Define the database directory
DB_DIRECTORY = "objectbox_db2"

# Ensure the directory exists
if not os.path.exists(DB_DIRECTORY):
    os.makedirs(DB_DIRECTORY)


# Define the entity class
@Entity()
class EmbeddingEntity:
    id = Id  # Primary key
    text = String  # Text content
    vector = Bytes  # Serialized embedding vector
    metadata = String  # Metadata (JSON as string)


# Create and configure the model
model = Model()
model.entity(EmbeddingEntity)  # Register the entity in the model

# store = Store(directory=DB_DIRECTORY)
# Initialize ObjectBox Store

print("ObjectBox initialized successfully!")

# Paths and Constants
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


# Embedding function
def get_embedding_function():
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


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
        loader = WebBaseLoader(web_url=url)
        async for doc in loader.alazy_load():
            combined_text += doc.page_content + "\n"
    return combined_text


# Split text into chunks
def split_text(text):
    splitter = SemanticChunker(get_embedding_function())
    return splitter.split_text(text)


# Add chunks to ObjectBox
def add_to_objectbox(chunks):
    embedding_function = get_embedding_function()

    # Get the ObjectBox box for EmbeddingEntity
    box = objectbox.box_for(EmbeddingEntity)

    for i, chunk in enumerate(chunks):
        # Generate embedding for the chunk
        embedding = embedding_function.embed_query(chunk)

        # Create new entity
        entity = EmbeddingEntity(
            text=chunk,
            vector=pickle.dumps(embedding),  # Serialize the numpy array
            metadata=f"chunk_{i}"  # Store metadata
        )

        # Put entity in the database
        box.put(entity)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Query ObjectBox with similarity search
def query_objectbox(query_text, k=5):
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query_text)

    box = objectbox.box_for(EmbeddingEntity)
    all_entities = box.get_all()

    # Calculate similarities
    similarities = []
    for entity in all_entities:
        stored_embedding = pickle.loads(entity.vector)
        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((entity, similarity))

    # Sort by similarity and get top k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# Initialize database endpoint
@app.route("/initialize", methods=["POST"])
async def initialize():
    pdf_text = load_pdfs(PDF_FILES)
    web_text = await load_web_pages(WEB_URLS)
    combined_text = pdf_text + web_text
    chunks = split_text(combined_text)
    add_to_objectbox(chunks)
    return jsonify({"message": "Database initialized with PDFs and web data successfully!"})


# Query endpoint
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    # Get similar documents
    results = query_objectbox(query_text)
    context_text = "\n\n---\n\n".join([entity.text for entity, _ in results])

    # Prepare and run the prompt
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:

    {context}

    ---

    Question: {question}
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = HuggingFaceEndpoint(
            OLLAMA_URL,
            temperature=0.7,
            model_kwargs={
                "api_key": "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",
                "max_length": 512,
            }
        )
    response_text = model.invoke(prompt)
    sources = [entity.metadata for entity, _ in results]

    return {"response": response_text, "sources": sources}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query text for RAG.", required=True)
    args = parser.parse_args()

    if not os.path.exists(DB_DIRECTORY):
        print("Initializing database...")
        pdf_text = load_pdfs(PDF_FILES)
        web_text = await load_web_pages(WEB_URLS)
        combined_text = pdf_text + web_text
        chunks = split_text(combined_text)
        add_to_objectbox(chunks)
        print("Database initialized!")

    print("Querying...")
    result = query_objectbox(args.query)
    print(result)


if __name__ == "__main__":
    app.run(debug=True)
