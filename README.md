# Research Q&A Chatbot

The **Research Q&A Chatbot** is an AI-powered tool that enables users to ask complex questions related to the contents of uploaded PDF documents. Built using a **Retrieval-Augmented Generation (RAG)** framework, it leverages advanced language models and embeddings to generate accurate, contextually relevant answers, along with citations from the referenced documents.

## Key Features

- **Upload and process up to 10 PDF files**: The chatbot can handle multiple research papers or documents, processing them into a searchable format.
- **Context-aware answers**: Provides accurate responses to user queries by analyzing the contents of the uploaded documents.
- **Citations included**: All responses are accompanied by citations, referencing the specific document(s) from which the answers were derived.
- **Powered by Google Generative AI**: Utilizes **Google's Generative AI models (Gemini-Pro)** for intelligent question answering and embeddings.
- **Built with Streamlit**: A user-friendly web interface to easily upload documents and ask questions.

## How It Works

1. **Upload Documents**: Users can upload up to 10 PDF files at a time.
2. **Processing**: The documents are broken down into manageable chunks, and embeddings are generated to create a vector store for efficient search and retrieval.
3. **Ask a Question**: Users can input any query, and the bot will search the content of all uploaded files to provide a detailed, accurate answer with relevant citations.

## Technologies Used

- **Streamlit**: Provides an interactive web interface for document uploading and querying.
- **LangChain**: Handles the text processing pipeline, including document chunking, embeddings, and chaining the model with a prompt template for response generation.
  - **LangChain Google Generative AI**: Used for embeddings and conversational response generation.
- **Google's Gemini-Pro**: A large language model used for high-quality, context-aware responses.
- **FAISS (Facebook AI Similarity Search)**: An efficient vector store to handle similarity searches over document chunks.
- **PyPDF2**: Extracts text from the uploaded PDF documents.
  
## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Sanjana2306/Research_QA_Bot.git
    cd Research_QA_Bot
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Obtain Google API Key**:
    - Visit [Google MakerSuite](https://makersuite.google.com/app/apikey) to generate an API key.
    - Add the key when prompted by the app.

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run streamlit_App.py
    ```
    
2. **Upload PDFs**: In the sidebar, upload up to 10 PDF documents.

3. **Ask a question**: Once the documents are processed, input a question in the main interface, and the chatbot will return relevant answers with citations from the uploaded PDFs.

## System Requirements

- **Python 3.8 or later**
- **Google API Key** (for access to the generative AI models)

## Future Improvements

- Add support for other document formats (e.g., DOCX, HTML).
- Improve question understanding using custom fine-tuning of the generative AI models.
- Implement caching for faster repeated queries.

