import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("📄 AI-Powered Financial Report Assistant")

# === Initialize session state variables ===
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "New PDF"
if "uploaded_pdf" not in st.session_state:
    st.session_state["uploaded_pdf"] = None
if "s3_markdown_path" not in st.session_state:
    st.session_state["s3_markdown_path"] = None
if "selected_rag_method" not in st.session_state:
    st.session_state["selected_rag_method"] = "Select a RAG Method"
if "selected_chunking" not in st.session_state:
    st.session_state["selected_chunking"] = "Select a Chunking Strategy"
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "query_response" not in st.session_state:
    st.session_state["query_response"] = ""
if "question_input_key" not in st.session_state:
    st.session_state["question_input_key"] = 0  # Key for forcing UI refresh

# === Function to Reset State on PDF Selection, Upload, or Tab Switch ===
def reset_session():
    """Reset session variables when a new PDF is selected, uploaded, or when switching tabs."""
    st.session_state["selected_rag_method"] = "Select a RAG Method"
    st.session_state["selected_chunking"] = "Select a Chunking Strategy"
    st.session_state["user_query"] = ""
    st.session_state["query_response"] = ""
    st.session_state["question_input_key"] += 1  # Force UI refresh of text input

# === Tabs for Uploading & Selecting PDFs ===
selected_tab = st.radio("Select an option:", ["New PDF", "Processed PDF"], horizontal=True)

# Reset session state when switching tabs
if selected_tab != st.session_state["active_tab"]:
    st.session_state["active_tab"] = selected_tab
    reset_session()  # Ensures everything is cleared when switching tabs

# === Tab 1: Upload a New PDF ===
if selected_tab == "New PDF":
    st.markdown("#### Upload New PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload", on_change=reset_session)

    if uploaded_file and st.button("Process PDF", on_click=reset_session):
        st.write("Processing your PDF...")

        files = {"file": uploaded_file}
        response = requests.post(f"{FASTAPI_URL}/upload_pdf/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"PDF processed successfully!")
            st.session_state["uploaded_pdf"] = data['pdf_filename']
            st.session_state["s3_markdown_path"] = data['markdown_s3_url']
        else:
            st.error(f"Failed to process PDF! Error: {response.text}")

# === Tab 2: Select a Processed PDF ===
elif selected_tab == "Processed PDF":
    st.markdown("#### Select a Previously Processed PDF")

    response = requests.get(f"{FASTAPI_URL}/select_pdfcontent/")
    if response.status_code == 200:
        processed_pdfs = response.json().get("processed_pdfs", {})
        pdf_options = ["Select a Processed PDF"] + list(processed_pdfs.keys())

        # Default to uploaded PDF if available
        default_index = pdf_options.index(st.session_state["uploaded_pdf"]) if st.session_state["uploaded_pdf"] in pdf_options else 0
        selected_pdf = st.selectbox(
            "Choose a PDF",
            pdf_options,
            index=default_index,
            key="processed_pdf_select",
            on_change=reset_session
        )

        if selected_pdf != "Select a Processed PDF":
            st.session_state["uploaded_pdf"] = selected_pdf
            st.session_state["s3_markdown_path"] = processed_pdfs[selected_pdf]["markdown"]

            if st.session_state["s3_markdown_path"]:
                if st.button("View Markdown"):
                    markdown_content = requests.get(st.session_state["s3_markdown_path"]).text
                    st.text_area("Markdown Preview", markdown_content, height=400, disabled=True)

                st.download_button("Download Markdown", st.session_state["s3_markdown_path"], file_name=f"{selected_pdf}.md")

# === Configure RAG Retrieval ===
st.markdown("### Configure RAG Retrieval")

rag_methods = ["Select a RAG Method", "Manual Embeddings", "Pinecone", "ChromaDB"]
st.session_state["selected_rag_method"] = st.selectbox(
    "Choose RAG Method",
    rag_methods,
    index=rag_methods.index(st.session_state["selected_rag_method"]),
    key="rag_method",
    on_change=reset_session
)

chunking_strategies = ["Select a Chunking Strategy", "Cluster-based", "Token-based", "Recursive-based"]
st.session_state["selected_chunking"] = st.selectbox(
    "Choose Chunking Strategy",
    chunking_strategies,
    index=chunking_strategies.index(st.session_state["selected_chunking"]),
    key="chunking_method",
    on_change=reset_session
)


# === Query Input and Retrieval ===
st.markdown("### Ask a Question")

user_query = st.text_input(
    "Ask a question about the document:",
    value=st.session_state["user_query"],
    key=f"question_input_{st.session_state['question_input_key']}"
)

if st.button("🔍 Retrieve Answer"):
    if not st.session_state["uploaded_pdf"]:
        st.error("No PDF selected. Please upload or choose a processed PDF.")
    elif st.session_state["selected_rag_method"] == "Select a RAG Method":
        st.error("Please select a RAG Method.")
    elif st.session_state["selected_chunking"] == "Select a Chunking Strategy":
        st.error("Please select a Chunking Strategy.")
    elif not user_query:
        st.error("Please enter a query.")
    else:
        query_payload = {
            "pdf_name": st.session_state["uploaded_pdf"],
            "query": user_query,
            "rag_method": st.session_state["selected_rag_method"],
            "chunking_strategy": st.session_state["selected_chunking"],
            "s3_markdown_path": st.session_state["s3_markdown_path"],
            "top_k": 3
        }
        response = requests.post(f"{FASTAPI_URL}/query/", json=query_payload)

        if response.status_code == 200:
            result = response.json()["response"]
            st.success("Answer Retrieved:")
            st.text_area("Query Response", result, height=300, disabled=True)
        else:
            st.error(f"Failed to retrieve an answer: {response.json().get('error', 'Unknown error')}")