# MultiPDF Chat App

## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively. The text chunks are then converted to vectors by [HuggingFaceEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceEmbeddings.html) using the model [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). The vectors are then stored in Pinecone vector database.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks. In this sample I am using Llama 2 13B chat model from [TheBloke/Llama-2-13B-chat-GGML](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML). The model name is llama-2-13b-chat.ggmlv3.q5_1.bin.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

Optain these info for authorization:
- Huggingface API Token: needed when you want to download the LLM from Huggingface at run time. If you choose to pre-download, store and load the model locally, you don't need this token 
- Pinecone API Key: use for Pinecone auth
- Pinecone API Env: Pinecone environment where the Index was created

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added these info to the `.env` file.
- HUGGINGFACEHUB_API_TOKEN=<your Huggingface api token>
- PINECONE_API_KEY=<your Pinecone api key>
- PINECONE_API_ENV=<your Pinecone API>

Open the app.py. Look for the 

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Known Limitations
If you're using Starter (Free Tier) Pinecone vector datababase, the Starter Index limits the amount of vectors being stored. Recommended to use small size PDFs to try out.
If this is the first time running, the application will try to create a new Index in Pinecone. The index name is picked up from PINECONE_INDEX_NAME var specified in `.env` file. The new index initialization might take some time to complete in Pinecone, so check your Pinecone for index initialization status *before* uploading Pdfs via the UI.
