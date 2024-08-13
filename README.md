# RAG with LLAMA3, GROQ Inferencing Engine, and ObjectBox

This project demonstrates a Retrieval-Augmented Generation (RAG) system using LLAMA3, GROQ Inferencing Engine, and ObjectBox. It allows users to upload PDF documents, create embeddings for the text, and then retrieve answers to queries based on the contents of those documents.

## Features

- **Upload PDFs**: Upload multiple PDF documents, which are then stored in the `docs` directory.
- **Document Management**: View the list of uploaded documents or clear them from the storage.
- **Text Embedding and Vector Store**: Convert the text content of PDFs into embeddings using `OllamaEmbeddings` and store them in an ObjectBox vector store.
- **Querying**: Input a query to retrieve answers based on the context within the uploaded PDFs using LLAMA3 via GROQ Inferencing Engine.
- **Contextual Source Display**: View the exact sources of the retrieved answers.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:

    Create a `.env` file in the root directory and add your GROQ API key:

    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload PDFs**: Click on "Choose PDF files" to upload your documents.
2. **Check Present Files**: Verify the uploaded files by clicking "Check Present files."
3. **Clear Uploaded Content**: Clear all uploaded PDFs from the storage by clicking "Clear Uploaded content."
4. **Use Data from Current PDFs**: Click the "Use data from current pdfs" button to process the PDFs and create embeddings.
5. **Enter a Query**: Use the text input to enter a prompt and retrieve an answer from the documents.
6. **View Context**: Expand the "Check source of answer" section to see the document excerpts used to generate the response.

## Results

https://github.com/user-attachments/assets/856a9cdc-a7e5-43f2-8f7a-590ca74ebb4c

https://github.com/user-attachments/assets/66705f8c-1bf3-4510-9a36-9ec5b101a9ae

## Contributing

Feel free to open issues or submit pull requests for new features or bug fixes.


## Acknowledgments

- **Streamlit**: For the easy-to-use web app framework.
- **Langchain**: For providing the integration with various LLMs and vector stores.
- **GROQ**: For the high-performance inferencing engine.
- **ObjectBox**: For the efficient vector storage and retrieval.

