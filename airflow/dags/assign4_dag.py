from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import boto3
import os
import pickle

# Import custom processing functions
from data_processing.pdf_extract_mistral import process_pdf_mistral
from data_processing.pinecone_rag import pinecone_rag_airflow
from data_processing.pdf_extract_docling import process_pdf_docling
from data_processing.s3_utils import generate_presigned_url, fetch_markdown_from_s3
from data_processing.chroma_rag_pipeline import chroma_rag_airflow
from data_processing.naive_rag import naive_embedding_airflow

# Airflow imports for DAG creation
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

# ---------- S3 Helper Functions ----------

def get_s3_client():
    """Initialize and return an S3 client and bucket name from environment variables."""
    load_dotenv()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    return s3_client, os.getenv("S3_BUCKET_NAME")

def list_files_in_bucket(bucket_name, prefix=''):
    """
    List all files (object keys) in the specified S3 bucket, optionally filtered by prefix.
    """
    s3_client, _ = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")
    file_list = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_list.append(obj['Key'])
    return file_list

def download_pdf_files(bucket_name, prefix='', download_dir=None):
    """
    Downloads all PDF files from the specified S3 bucket (and prefix) into a local directory.
    Each file is saved with its base name.
    """
    if download_dir is None:
        download_dir = os.path.join("/opt/airflow/dags/", "tmp")
        print("Download directory:", download_dir)
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    
    files = list_files_in_bucket(bucket_name, prefix=prefix)
    local_files = []
    
    # Filter and download only PDF files
    for file_key in files:
        if file_key.lower().endswith(".pdf"):
            filename = os.path.basename(file_key)
            local_path = os.path.join(download_dir, filename)
            try:
                s3_client, _ = get_s3_client()
                s3_client.download_file(bucket_name, file_key, local_path)
                print(f"Downloaded {file_key} to {local_path}")
                local_files.append(local_path)
            except Exception as e:
                print(f"Error downloading {file_key}: {e}")
    
    return local_files

# ---------- DAG Definition and Tasks ----------

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG with selectable parameters via the Airflow UI
dag = DAG(
    dag_id='Assignment4.2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    params={
        "pdf_processor": Param(
            "Docling", 
            type="string", 
            description="Select PDF Processor: Docling or Mistral_OCR",
            title="PDF Processor"
        ),
        "rag_pipeline": Param(
            "Manual_Embedding",
            type="string",
            description="Select RAG Pipeline: Manual_Embedding, Pinecone or ChromaDB",
            title="RAG Pipeline"
        ),
        "chunking_strategy": Param(
            "Recursive_Token",
            type="string",
            description="Select Chunking Strategy: Cluster, Token or Recursive",
            title="Chunking Strategy"
        ),
    },
)

def process_parameters(**kwargs):
    """
    Retrieve and log user-supplied parameters from the Airflow UI.
    """
    params = kwargs['params']
    pdf_processor = params.get('pdf_processor')
    rag_pipeline = params.get('rag_pipeline')
    chunking_strategy = params.get('chunking_strategy')
    
    print(f"Selected PDF Processor: {pdf_processor}")
    print(f"Selected RAG Pipeline: {rag_pipeline}")
    print(f"Selected Chunking Strategy: {chunking_strategy}")

def download_pdf_files_task(**kwargs):
    """
    Downloads PDF files from the configured S3 bucket.
    """
    _, bucket_name = get_s3_client()
    prefix = ''  # Adjust if your PDF files are stored under a folder in S3
    downloaded_pdfs = download_pdf_files(bucket_name=bucket_name, prefix=prefix)
    print("Downloaded PDF files:", downloaded_pdfs)
    return downloaded_pdfs  # XCom: list of local PDF file paths

def process_pdf_file_task(**kwargs):
    """
    Processes all PDF files from the download directory using the selected PDF processor.
    Returns a dictionary mapping file names to their corresponding S3 markdown URLs.
    """
    params = kwargs['params']
    pdf_processor = params.get('pdf_processor')
    print(f"Processing files using {pdf_processor} processor.")
    
    download_dir = os.path.join("/opt/airflow/dags/", "tmp")
    
    # List all PDF files in the download directory
    pdf_files = [f for f in os.listdir(download_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found for processing.")
        return None
        
    files_to_process = pdf_files[:2]
    results = {}
    for file_name in files_to_process:
        file_path = os.path.join(download_dir, file_name)
        print(f"Processing file: {file_path}")
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        if pdf_processor.lower() == "docling":
            s3_markdown_url = process_pdf_docling(file_content, file_name)
        else:
            s3_markdown_url = process_pdf_mistral(file_content, file_name)
            time.sleep(1)
        print(f"S3 Markdown URL for {file_name}: {s3_markdown_url}")
        results[file_name] = {"markdown_s3_url": s3_markdown_url}
    
    # Return a dict with the markdown URLs for further processing
    return {"markdown_results": results}

def chunking_embedding_task(**kwargs):
    """
    Retrieves the dictionary of S3 markdown URLs from the previous task.
    For each file, it:
      - Generates a presigned URL,
      - Fetches the markdown content from S3, and
      - Passes the content to the appropriate embedding function.
    
    For Manual_Embedding, the embeddings dictionary is pickled to a local file and
    the file path is returned. For other pipelines, the embedding results are returned directly.
    """
    import pickle  # Ensure pickle is imported here
    ti = kwargs['ti']
    params = kwargs['params']
    rag_pipeline = params.get('rag_pipeline')
    
    result_dict = ti.xcom_pull(task_ids='process_pdf_file')
    if not result_dict or "markdown_results" not in result_dict:
        print("No S3 markdown URLs received.")
        return
    markdown_results = result_dict["markdown_results"]
    print("Markdown results from process_pdf_file:", markdown_results)
    
    embedding_results = {}
    for file_name, urls in markdown_results.items():
        # Extract the nested URL dictionary and then the actual URL string
        url_dict = urls['markdown_s3_url']
        actual_url = url_dict['markdown_s3_url']
        
        # Extract S3 key from the URL (assuming format "https://bucket.s3.amazonaws.com/path/to/file")
        if ".com/" in actual_url:
            key = actual_url.split('.com/')[1]
        else:
            key = actual_url.split("s3://")[1].split("/", 1)[1]
        
        presigned_url = generate_presigned_url(key)
        print(f"Generated presigned URL for {file_name}: {presigned_url}")
        markdown_content = fetch_markdown_from_s3(key)
        print(f"Fetched markdown content for {file_name}: {markdown_content}")
        
        # Call the appropriate embedding function based on the selected pipeline
        if rag_pipeline.lower() == "manual_embedding":
            embedding_result = naive_embedding_airflow(markdown_content, "Cluster-based")
        elif rag_pipeline.lower() == "pinecone":
            embedding_result = pinecone_rag_airflow(presigned_url, markdown_content, "Token-based")
        else:   
            embedding_result = chroma_rag_airflow(presigned_url, markdown_content, "Recursive-based")
        
        print(f"Embedding result for {file_name}: {embedding_result}")
        embedding_results[file_name] = embedding_result

    # If Manual_Embedding is selected, dump the embeddings dict to a pickle file and return the path.
    if rag_pipeline.lower() == "manual_embedding":
        local_pkl_path = os.path.join("/opt/airflow/dags/", "manual_embedding.pkl")
        with open(local_pkl_path, 'wb') as fp:
            pickle.dump(embedding_results, fp)
        print(f"Embeddings stored in pickle file: {local_pkl_path}")
        return {"manual_embedding_path": local_pkl_path}
    else:
        # For other pipelines, return the embeddings dictionary directly.
        return embedding_results


def store_manual_embedding_task(**kwargs):
    """
    Next task that processes the output of the chunking_embedding_task.
    If the output contains a 'manual_embedding_path', it loads the pickle file
    and then uploads it to S3. Otherwise, it does nothing.
    """
    import pickle
    ti = kwargs['ti']
    result = ti.xcom_pull(task_ids='chunking_embedding')
    
    # Check if the manual embedding file was produced
    if result and "manual_embedding_path" in result:
        local_pkl_path = result["manual_embedding_path"]
        print(f"Manual embedding pickle file found at: {local_pkl_path}")
        
        # Load the embeddings from the pickle file
        with open(local_pkl_path, "rb") as fp:
            embeddings = pickle.load(fp)
        print("Loaded embeddings:", embeddings)
        
        # Upload the pickle file to S3 (if desired)
        s3_client, bucket_name = get_s3_client()
        s3_key = "final_embeddings/embeddings.pkl"
        try:
            s3_client.upload_file(local_pkl_path, bucket_name, s3_key)
            s3_url = f"s3://{bucket_name}/{s3_key}"
            print("Uploaded manual embeddings to S3:", s3_url)
            return {"final_manual_embedding_s3_url": s3_url}
        except Exception as e:
            print(f"Error uploading the pickle file to S3: {e}")
            return None
    else:
        print("No manual embedding file produced; skipping manual embedding processing.")
        return None

# ---------- DAG Task Definitions ----------

select_parameters_task = PythonOperator(
    task_id='select_parameters',
    python_callable=process_parameters,
    provide_context=True,
    dag=dag,
)

download_pdf_files_operator = PythonOperator(
    task_id='download_pdf_files',
    python_callable=download_pdf_files_task,
    provide_context=True,
    dag=dag,
)

process_pdf_file_operator = PythonOperator(
    task_id='process_pdf_file',
    python_callable=process_pdf_file_task,
    provide_context=True,
    dag=dag,
)

chunking_embedding_operator = PythonOperator(
    task_id='chunking_embedding',
    python_callable=chunking_embedding_task,
    provide_context=True,
    dag=dag,
)

store_manual_embedding_operator = PythonOperator(
    task_id='store_manual_embedding',
    python_callable=store_manual_embedding_task,
    provide_context=True,
    dag=dag,
)

# ---------- Task Dependencies ----------
select_parameters_task >> download_pdf_files_operator >> process_pdf_file_operator >> chunking_embedding_operator >> store_manual_embedding_operator