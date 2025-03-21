import os
import requests
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv
from data_processing.chunking import embedding_model, cluster_based_chunking, recursive_based_chunking, token_based_chunking
import inspect

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)


def get_or_create_index():
    """
    Ensure that the specified Pinecone index exists and return the index object.
    """
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        print(f"Created new index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)


def add_chunks_to_pinecone(chunks, markdown_file_path):
    """
    Add text chunks and their embeddings to the Pinecone index.
    """
    index = get_or_create_index()
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    ids = [f"{markdown_file_path}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": markdown_file_path, "chunk_index": i, "text": chunks[i]} for i in range(len(chunks))]

    index.upsert(vectors=[(ids[i], embeddings[i].tolist(), metadatas[i]) for i in range(len(chunks))])
    return len(chunks)


def retrieve_relevant_chunks(query, metadata_filter=None, top_k=5):
    """
    Perform strict retrieval with metadata filtering.
    """
    index = get_or_create_index()
    query_embedding = embed_texts([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter  
    )

    chunks, sources = [], []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                chunks.append(match["metadata"]["text"])
                sources.append(match["metadata"]["source"])

    return chunks, sources


def generate_response(query, chunks, sources):
    """
    Generate a response strictly from the retrieved document chunks.
    """
    openai.api_key=OPENAI_API_KEY
    
    if not chunks:
        return "No relevant information found."

    context = "\n\n".join([f"Source [{i+1}] ({source}): {chunk}" for i, (chunk, source) in enumerate(zip(chunks, sources))])

    system_message = "You are an assistant that only provides answers based on the provided document context."

    user_message = f"Question: {query}\n\nContext:\n{context}\n\nAnswer based strictly on the provided context."

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
    

def pinecone_rag_pipeline(s3_markdown_path, query, chunking_strategy,top_k):
    """
    Run the RAG pipeline while ensuring that answers are only retrieved from the uploaded document.
    """
    response = requests.get(s3_markdown_path)
    
    if response.status_code != 200:
        return "Failed to retrieve Markdown from S3."

    document_text = response.text

    chunking_methods = {
        "Cluster-based": cluster_based_chunking,
        "Token-based": token_based_chunking,
        "Recursive-based": recursive_based_chunking
    }

    if chunking_strategy not in chunking_methods:
        return "Invalid chunking strategy selected."

    chunking_function = chunking_methods[chunking_strategy]
    if "max_chunk_size" in inspect.signature(chunking_function).parameters:
        chunks = chunking_function(document_text, max_chunk_size=300)
    else:
        chunks = chunking_function(document_text)

    add_chunks_to_pinecone(chunks, s3_markdown_path)

    
    retrieved_chunks, sources = retrieve_relevant_chunks(query, metadata_filter={"source": {"$eq": s3_markdown_path}}, top_k=top_k)

    
    return generate_response(query, retrieved_chunks, sources)
