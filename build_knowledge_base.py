import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# Load the profile document
with open("profile.txt", "r") as f:
    text = f.read()

# Split into chunks by paragraph - each blank line becomes a split point
# strip() removes extra whitespace, and the if-check skips empty chunks
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# Add chunks that match "What is my name?" and "Who are you?" queries.
# Profile uses "ABOUT" section; first chunk has "Name: X. I am a ..."
first_chunk = (chunks[0] if chunks else "").strip()
if first_chunk.upper().startswith("ABOUT"):
    lines = first_chunk.split("\n")
    name_line = next((l for l in lines if l.strip().lower().startswith("name:")), "")
    if name_line:
        # "Name: Pranay Bhasker Reddy. I am a B.Tech ..." -> use full sentence for "who are you"
        who_chunk = f"Who is this person: {name_line.strip()}"
        chunks = [who_chunk] + chunks
elif text.strip().split("\n")[0].strip().lower().startswith("name"):
    identity_chunk = f"Profile owner's identity: {text.strip().split(chr(10))[0].strip()}"
    chunks = [identity_chunk] + chunks

print(f"Loaded {len(chunks)} chunks from profile.txt")

# Initialize ChromaDB - PersistentClient saves data to disk so it survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

# Connect to Ollama's embedding model to convert text into vectors
ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",  # Ollama's default local address
)

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,  # Tells ChromaDB how to convert text to vectors
)

# Clear existing chunks so we can rebuild with updated chunking (e.g. after adding identity chunk)
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

# Add chunks to the collection - ChromaDB automatically generates embeddings
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],  # Unique ID for each chunk
    documents=chunks,  # The actual text content
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunks))],
)

print(f"Added {len(chunks)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")
