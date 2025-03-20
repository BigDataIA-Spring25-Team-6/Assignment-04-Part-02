import os
import json
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load OpenAI API key (if available)
api_key = os.environ.get("OPENAI_API_KEY")

# Load Sentence Transformer model (cached)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="huggingface_cache")

# Function to tokenize sentences
def tokenize_sentences(text):
    """Regex-based sentence tokenizer (No NLTK)."""
    return re.split(r'(?<=[.!?])\s+', text)  # Splits on punctuation marks

# Function to compute embeddings
def compute_embeddings(sentences):
    """Compute embeddings for a list of sentences."""
    return embedding_model.encode(sentences, convert_to_numpy=True)

# OpenAI token count function (Approximation)
def openai_token_count(text):
    return len(text.split())

# Function to analyze chunk statistics
def analyze_chunks_stats(chunks):
    """Analyze chunk size distributions."""
    stats = {
        "num_chunks": len(chunks),
        "avg_size_chars": sum(len(c) for c in chunks) / len(chunks),
        "avg_size_tokens": sum(openai_token_count(c) for c in chunks) / len(chunks),
        "min_size_tokens": min(openai_token_count(c) for c in chunks),
        "max_size_tokens": max(openai_token_count(c) for c in chunks)
    }
    return stats

# Function to save chunks to JSON
def save_chunks_to_json(chunks, strategy_name):
    """Save chunks to JSON for inspection."""
    os.makedirs("test", exist_ok=True)
    with open(f"test/{strategy_name}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
    print(f"Chunks saved to test/{strategy_name}_chunks.json")

# Main function for Cluster-Based Chunking
def cluster_based_chunking(document, max_chunk_size=300, similarity_threshold=0.75):
    """
    Perform semantic chunking by grouping sentences with high similarity.

    Args:
        document (str): The input text.
        max_chunk_size (int): Maximum chunk size in tokens.
        similarity_threshold (float): Threshold to merge sentences into chunks.

    Returns:
        list: List of clustered semantic chunks.
    """
    sentences = tokenize_sentences(document)  # Tokenize into sentences
    embeddings = compute_embeddings(sentences)  # Compute embeddings
    
    clusters = []
    current_chunk = []
    
    for i, sentence in enumerate(sentences):
        if i == 0:
            current_chunk.append(sentence)
            continue
        
        # Compare similarity with the last sentence in the chunk
        similarity = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]

        if similarity > similarity_threshold:
            current_chunk.append(sentence)  # Merge sentence into existing chunk
        else:
            clusters.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Start new chunk

    if current_chunk:
        clusters.append(" ".join(current_chunk))

    return clusters