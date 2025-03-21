from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from data_processing.naive_rag import naive_rag_pipeline
from data_processing.pinecone_rag import pinecone_rag_pipeline
from data_processing.chroma_rag_pipeline import chroma_rag_pipeline
from data_processing.s3_utils import s3_client, S3_BUCKET_NAME,generate_presigned_url
from data_processing.pdf_extract_docling import process_pdf_docling
from data_processing.pdf_extract_mistral import process_pdf_mistral
from data_processing.chunking import cluster_based_chunking, token_based_chunking,recursive_based_chunking

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    pdf_name: str
    query: str
    rag_method: str
    chunking_strategy: str
    s3_markdown_path: str = None
    top_k: int = 5

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """
    Handles user queries by dynamically selecting RAG method and chunking strategy.
    Automatically fetches the Markdown S3 path instead of requiring it in the request.
    """
   
    pdf_name = request.pdf_name
    query = request.query
    rag_method = request.rag_method
    chunking_strategy = request.chunking_strategy
    s3_markdown_path = request.s3_markdown_path

    if not s3_markdown_path:
        try:
            s3_list_response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{pdf_name}/markdown/")
            if "Contents" in s3_list_response and s3_list_response["Contents"]:
                s3_markdown_path = s3_list_response["Contents"][0]["Key"]
            else:
                return JSONResponse(content={"error": f"Markdown file for {pdf_name} not found in S3."}, status_code=404)
        except Exception as e:
            return JSONResponse(content={"error": f"Error fetching Markdown from S3: {str(e)}"}, status_code=500)
        
    # Select Chunking Method
    chunking_methods = {
        "Cluster-based": cluster_based_chunking,
        "Token-based": token_based_chunking,
        "Recursive-based": recursive_based_chunking
    }

    
    if chunking_strategy not in chunking_methods:
        return JSONResponse(content={"error": "Invalid chunking strategy selected."}, status_code=400)

    # Select RAG Method (only Naïve RAG for now, Pinecone/ChromaDB future-proofed)
    rag_methods = {
        "Manual Embeddings": naive_rag_pipeline,
        "Pinecone": pinecone_rag_pipeline,
        "ChromaDB": chroma_rag_pipeline
    }
    if rag_method not in rag_methods:
        return JSONResponse(content={"error": f"Invalid RAG method: {rag_method}"}, status_code=400)

    # Execute the RAG pipeline with the selected chunking method
    response = rag_methods[rag_method](s3_markdown_path, query, chunking_strategy, top_k=request.top_k)

    return {"response": response, "s3_markdown_url": s3_markdown_path}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), parser: str = Form(...)):
    """
    Handles PDF file upload, processes it, and stores results in S3.
    """
    
    # Mapping of parsing methods
    parsing_methods = {
        "Docling": process_pdf_docling,
        "Mistral": process_pdf_mistral
    }

    try:
        # Validate parser selection
        if parser not in parsing_methods:
            return JSONResponse(content={"error": "Invalid parsing method selected."}, status_code=400)

        # Read the uploaded file content
        file_content = await file.read()

        # Call the selected parser function dynamically
        result = parsing_methods[parser](file_content, file.filename)
        markdown_s3_url = result["markdown_s3_url"]
        s3_base_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/"
        if markdown_s3_url.startswith(s3_base_url):
            object_key = markdown_s3_url.replace(s3_base_url, "")
            signed_url = generate_presigned_url(object_key)
            if not signed_url:
                raise HTTPException(status_code=500, detail="Failed to generate signed URL.")
            markdown_s3_url = signed_url

        # Step 3: Return the S3 URLs and other details
        return {
            "message": result["message"],
            "markdown_s3_url": markdown_s3_url,  # Markdown URL
            "image_s3_urls": result["image_s3_urls"],      # List of Image URLs
            "pdf_filename": result["pdf_filename"],        # Filename without extension (used for grouping)
            "status": result["status"]
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/select_pdfcontent/")
async def select_pdfcontent():
    """
    Lists all previously processed PDFs stored in S3 with pre-signed URLs.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        pdf_files = {}

        if "Contents" in response:
            for obj in response["Contents"]:
                object_key = obj["Key"]
                parts = object_key.split("/")

                if len(parts) > 1:
                    pdf_name = parts[0]

                    if pdf_name not in pdf_files:
                        pdf_files[pdf_name] = {"markdown": None, "images": []}

                    if "markdown" in object_key:
                        pdf_files[pdf_name]["markdown"] = generate_presigned_url(object_key)
                    elif "images" in object_key:
                        pdf_files[pdf_name]["images"].append(generate_presigned_url(object_key))

        return {"processed_pdfs": pdf_files}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Run the FastAPI server locally on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)