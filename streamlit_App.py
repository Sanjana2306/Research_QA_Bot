import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Configure Streamlit app layout and title
st.set_page_config(page_title="Research Q&A Chatbot", layout="wide")

# Input for Google API Key
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Function to extract text from uploaded PDF documents and return with document names
def get_pdf_text(pdf_docs):
    doc_texts = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        doc_texts[pdf.name] = text
    return doc_texts

# Function to split text into chunks
def get_text_chunks(doc_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = []
    doc_metadata = []
    
    for doc_name, text in doc_texts.items():
        split_chunks = text_splitter.split_text(text)
        chunks.extend(split_chunks)
        doc_metadata.extend([doc_name] * len(split_chunks))  # Keep track of which document each chunk came from
    
    return chunks, doc_metadata

# Function to create and save vector store with document metadata for citations
def get_vector_store(text_chunks, doc_metadata, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=[{"doc": doc} for doc in doc_metadata])
    vector_store.save_local("faiss_index")

# Function to create conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question based on the provided context from the documents. If the answer is not in the context, respond with
    "The answer is not available in the provided context." Include citations by referring to the document name(s).\n\n
    Context:\n{context}\n
    Question: {question}\n
    Answer with citation:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and fetch answers with citations
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Extract the relevant context for answering
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Extract citations from metadata and display answers
    st.write("Reply: ", response["output_text"])
    st.write("Citations:")
    for doc in docs:
        st.write(f" - {doc.metadata['doc']}")

# Main function to handle user interaction
def main():
    st.header("Research Q&A Chatbot")

    user_question = st.text_input("Ask a question from the uploaded PDF files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Upload Documents:")
        pdf_docs = st.file_uploader("Upload up to 10 PDF Files", accept_multiple_files=True, key="pdf_uploader", type="pdf")
        
        # Limit to 10 PDF files
        if len(pdf_docs) > 10:
            st.error("You can upload a maximum of 10 PDF files.")
        elif st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                doc_texts = get_pdf_text(pdf_docs)
                text_chunks, doc_metadata = get_text_chunks(doc_texts)
                get_vector_store(text_chunks, doc_metadata, api_key)
                st.success("Processing complete")

if __name__ == "__main__":
    main()
