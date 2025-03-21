import openai
import os
import requests
from dotenv import load_dotenv
from data_processing.chunking import cluster_based_chunking
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="huggingface_cache")

def compute_and_store_embeddings(chunks):
    """
    Compute embeddings for each chunk and store them in memory.
    """
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return {"chunks": chunks, "embeddings": embeddings}

def retrieve_relevant_chunks(query, chunk_store, top_k=5, threshold=0.4):
    """
    Retrieve the most relevant chunks using cosine similarity.
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, chunk_store["embeddings"]).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    max_similarity = similarities[top_indices[0]]
    if max_similarity < threshold:
        return ["The query is not relevant to this document. Please refine your question."]

    return [chunk_store["chunks"][i] for i in top_indices]

def generate_llm_response(relevant_chunks, query):
    """
    Use GPT-4 to generate a response based on retrieved document chunks.
    """
    openai.api_key = OPENAI_API_KEY

    if not openai.api_key:
        return "OpenAI API key is missing. Please add it to your .env file."

    # Format retrieved chunks into prompt
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    You are an AI assistant that provides accurate and well-sourced answers from financial reports.

    Context from SEC financial reports:
    {context}

    Question: {query}

    Provide a concise and well-grounded response based on the above context.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a financial data assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI API Error: {e}"

def naive_rag_pipeline(s3_markdown_path, query, chunking_strategy, top_k=5):
    """
    Full RAG pipeline: Retrieve relevant chunks & generate GPT-4 response.
    """
    # Fetch Markdown content from S3
    response = requests.get(s3_markdown_path)
    
    if response.status_code != 200:
        return "Failed to retrieve Markdown from S3."

    document_text = response.text

    # Select chunking method
    chunking_methods = {
        "Cluster-based": cluster_based_chunking,
        # Future: "Sentence-based": sentence_based_chunking,
        # Future: "Fixed Length": fixed_length_chunking,
    }

    if chunking_strategy not in chunking_methods:
        return "Invalid chunking strategy selected."

    # Apply selected chunking strategy
    chunks = chunking_methods[chunking_strategy](document_text, max_chunk_size=300)

    # Compute and store embeddings
    chunk_store = compute_and_store_embeddings(chunks)

    # Retrieve relevant chunks based on query
    relevant_chunks = retrieve_relevant_chunks(query, chunk_store, top_k)

    # Generate response using GPT-4
    response = generate_llm_response(relevant_chunks, query)

    return response