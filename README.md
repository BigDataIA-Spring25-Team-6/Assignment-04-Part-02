# Streamlit Application with LLM Integration

## Team Members
- **Aditi Ashutosh Deodhar**  002279575  
- **Lenin Kumar Gorle**       002803806  
- **Poorvika Girish Babu**    002801388  

## Project Overview
### Problem Statement
The project aims to build an AI-powered information retrieval system that processes unstructured data like PDFs using a Retrieval-Augmented Generation (RAG) pipeline. It requires implementing automated data ingestion, PDF parsing, chunking strategies, vector-based and naive retrieval methods, and deploying a user interface for querying through FastAPI and Streamlit.

### Methodology
Refer to the codelabs document for a detailed explanation of the QuickStart.

### Scope
```
The project aims to develop an AI-powered Retrieval-Augmented Generation (RAG) system for processing and retrieving information from NVIDIA quarterly reports using various parsing, chunking, and retrieval methods.

-Automate data ingestion and processing with Apache Airflow.
-Implement multiple PDF parsing strategies, including Docling and Mistral OCR.
-Develop a RAG pipeline using manual embeddings, Pinecone, and ChromaDB.
-Build a Streamlit-based user interface for document uploads, queries, and retrieval selection.
-Deploy the system with Docker, separating Airflow orchestration from the Streamlit + FastAPI querying pipeline.
```

## Technologies Used
```
-Apache Airflow
-Docling
-Mistral OCR
-FastAPI
-Streamlit
-Pinecone
-ChromaDB
-Sentence Transformers
-Docker
-AWS S3
-OpenAI GPT
```
  

## Architecture Diagram
![image](https://github.com/user-attachments/assets/360ec6d9-2d02-4801-ab6b-fad9950ea999)


## Codelabs Documentation
(https://codelabs-preview.appspot.com/?file_id=1FBrWZohwD3lDmmtl6bf-iy8wjwyqE9AVYGAozQyiXOU#0)

## Hosted Applications links 
- Frontend : https://frontend-487006321216.us-central1.run.app
- Backend : https://backend-487006321216.us-central1.run.app

## Prerequisites
```
-Python 3.10+
-Docker installed and running
-Docker Compose installed
-AWS S3 bucket with credentials
-OpenAI API key
-Pinecone API key
-Streamlit installed
-FastAPI framework
-Apache Airflow setup
-Virtual environment
```
  

## Set Up the Environment
```sh
# Clone the repository
git clone https://github.com/BigDataIA-Spring25-Team-6/Assignment-04-Part-02.git
cd DAMG7245-Assignment-04-Part-02.git

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your AWS, Pinecone, and OpenAI credentials

# Run FastAPI backend (inside /api folder)
cd api
uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit frontend (in a new terminal, inside /frontend folder)
cd ../frontend
streamlit run streamlit_app.py

# Setup and start Airflow (inside /airflow folder)
cd ../airflow
# Initialize Airflow metadata DB
airflow db init
# Create a user (run once)
airflow users create --username admin --firstname admin --lastname user --role Admin --email admin@example.com --password admin
# Start Airflow web server and scheduler
airflow webserver --port 8080
airflow scheduler

# Optional: Run using Docker Compose from root directory
docker-compose up --build

```

## Project Structure

```

ASSIGNMENT-04-PART-02/

├── airflow/           # Airflow orchestration

├── api/               # FastAPI backend

├── data_processing/   # Data processing scripts (chunking, RAG, parsing, etc.)

├── frontend/          # Streamlit frontend

├── .dockerignore      # Docker ignore file

├── .gitignore         # Git ignore file

├── Dockerfile         # Dockerfile for the project

├── docker-compose.app.yml  # Docker Compose file for app deployment

├── docker-compose.yaml     # Main Docker Compose file

├── requirements.txt   # Dependencies file

```
 
