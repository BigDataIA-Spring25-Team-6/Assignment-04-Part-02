import os
import requests
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from data_processing.chunking import cluster_based_chunking,recursive_based_chunking,token_based_chunking

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ChromaDB settings
COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./chroma_db"


def get_or_create_collection():
    """
    Get or create a ChromaDB collection.
    
    Returns:
        chromadb.Collection: The ChromaDB collection
    """
    # Create the persistent client
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Set up the embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",cache_folder="huggingface_cache"
    )
    
    # Try to get the collection if it exists
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"Using existing collection: {COLLECTION_NAME}")
    except:
        # Create a new collection if it doesn't exist
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    
    return collection

def add_chunks_to_collection(chunks, markdown_file_path):
    """
    Add document chunks to the ChromaDB collection.
    
    Args:
        chunks (list): List of text chunks
        url (str): Source URL of the document
        
    Returns:
        int: Number of chunks added
    """
    # Get the collection
    collection = get_or_create_collection()
    
    # Skip if no chunks
    if not chunks:
        return 0
    
    # Create IDs for each chunk
    sanitized_path = markdown_file_path.replace("/", "_").replace(".", "_")
    ids = [f"chunk_{sanitized_path}_{i}" for i in range(len(chunks))]
    
    # Create metadata for each chunk
    metadatas = [{"source": sanitized_path, "chunk_index": i} for i in range(len(chunks))]
    
    # Add chunks to the collection
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    return len(chunks)


def retrieve_relevant_chunks(query,metadata_filter=None,distance_threshold=0.7,top_k=5):
    """
    Retrieve the most relevant document chunks for a query.
    
    Args:
        query (str): The query text
        
    Returns:
        list: List of relevant document chunks
        list: List of source URLs for each chunk
    """
    # Get the collection
    collection = get_or_create_collection()
    if metadata_filter and "source" in metadata_filter:
        metadata_filter["source"] = metadata_filter["source"].replace("/", "_").replace(".", "_")
    
    
    # Query the collection for similar chunks
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=metadata_filter
    )
    
    # Extract the retrieved chunks and their sources
    chunks = results["documents"][0]  # First list is for the first query
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Filtering the relevant chunks based on distance threshold
    relevant_chunks = []
    relevant_sources = []
    
    # Extract the sources
    sources = [meta["source"] for meta in metadatas]
    
    # Print retrieval information for debugging
    print(f"Retrieved {len(chunks)} chunks for query: '{query}'")
    for i, (chunk, source, distance) in enumerate(zip(chunks, sources, distances)):
        if distance <= distance_threshold:
            relevant_chunks.append(chunk)
            relevant_sources.append(source)
            print("-"*40)
            print(f"\nChunk {i+1} (Distance: {distance:.4f}, Source: {source}):")
            preview = chunk
            print(preview)
    
    return relevant_chunks, relevant_sources

def generate_response(query, chunks, sources):
    """
    Generate a response using OpenAI's API based on the query and retrieved chunks.
    
    Args:
        query (str): The user's query
        chunks (list): List of relevant document chunks
        sources (list): List of source URLs for each chunk
        
    Returns:
        str: The generated response
    """
    # Combine chunks with their sources for better attribution
    context_with_sources = []
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        context_with_sources.append(f"Source [{i+1}] ({source}): {chunk}")
    
    context = "\n\n".join(context_with_sources)
    
    # Define system message with instructions
    system_message = """You are a helpful assistant that provides accurate information based on the given context. 
    If the context doesn't contain relevant information to answer the question, acknowledge that and provide general information if possible.
    Always cite your sources by referring to the source numbers provided in brackets. Do not make up information."""
    
    # Define the user message with query and context
    user_message = f"""Question: {query}
    
    Context information:
    {context}
    
    Please answer the question based on the context information provided."""
    
    try:
        # Call the OpenAI API
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1000   # Limit response length
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

def chroma_rag_pipeline(s3_markdown_path, query, chunking_strategy,top_k):
    """
    Full Chroma RAG pipeline: Retrieve relevant chunks & generate GPT-4 response.
    """
    # Fetch Markdown content from S3
    response = requests.get(s3_markdown_path)
    
    if response.status_code != 200:
        return "Failed to retrieve Markdown from S3."

    document_text = response.text

    # Select chunking method
    chunking_methods = {
        "Cluster-based": cluster_based_chunking,
        "Token-based": token_based_chunking,
        "Recursive-based": recursive_based_chunking
        # Future: "Fixed Length": fixed_length_chunking,
    }

    if chunking_strategy not in chunking_methods:
        return "Invalid chunking strategy selected."

    # Apply selected chunking strategy
    chunks = chunking_methods[chunking_strategy](document_text, max_chunk_size=300)

    # Add chunks to the ChromaDB collection
    num_added = add_chunks_to_collection(chunks, s3_markdown_path)
    print(f"Added {num_added} chunks to vector store")

    # Retrieve relevant chunks
    print(f"\nStep 4: Retrieving relevant chunks for query: '{query}'")
    relevant_chunks, sources = retrieve_relevant_chunks(query, metadata_filter={"source": s3_markdown_path},top_k=top_k)

    # Generate response using OpenAI's GPT-4
    if relevant_chunks:
        response = generate_response(query, relevant_chunks, sources)
        return response
    else:
        return "No relevant information found to answer the query."

