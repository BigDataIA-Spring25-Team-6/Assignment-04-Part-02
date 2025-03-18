import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from data_processing.pdf_extract_docling import process_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handles PDF file upload, processes it, and stores results in S3.
    """
    try:
        # Step 1: Read the uploaded file content
        file_content = await file.read()

        # Step 2: Call process_pdf with original filename for structured S3 storage
        result = process_pdf(file_content, file.filename)

        # Step 3: Return the S3 URLs and other details
        return {
            "message": result["message"],
            "markdown_s3_url": result["markdown_s3_url"],  # Markdown URL
            "image_s3_urls": result["image_s3_urls"],      # List of Image URLs
            "pdf_filename": result["pdf_filename"],        # Filename without extension (used for grouping)
            "status": result["status"]
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI server locally on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)