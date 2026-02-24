from fastapi import FastAPI
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()  # Create the FastAPI application

# Connect to the same ChromaDB collection you built in Step 2
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)



@app.get("/ask")  # This creates a GET endpoint at /ask
def ask(question: str):  # FastAPI automatically reads "question" from the URL query string
    # Step 1: RETRIEVE - search ChromaDB for the most relevant chunks
    results = collection.query(
        query_texts=[question],  # ChromaDB converts this to a vector and finds similar chunks
        n_results=5,  # Retrieve more chunks so identity questions (e.g. "What is my name") get the right context
    )
    # Combine the matching chunks into a single string (filter out None if any)
    docs = results["documents"][0] or []
    context = "\n\n".join(d for d in docs if d)

    # Step 2: AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""The context below describes one specific person's profile. Answer the question about THAT person only. Do not describe yourself (the AI assistant); always describe the person in the context. If the context does not contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer along with the context so users can verify the source
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
