# RAG QnA Bot & LLM Validation

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) QnA Bot** alongside a **Large Language Model (LLM) Validation** system. The primary goal is to enhance the accuracy and reliability of responses by integrating an information retrieval system with an LLM, ensuring factually correct and contextually relevant answers.

## Features
- **Retrieval-Augmented Generation (RAG) QnA**: Retrieves relevant documents before generating answers using an LLM.
- **LLM Validation**: Cross-verifies responses to detect hallucinations and improve accuracy.
- **Contextual Awareness**: Enhances response quality by incorporating retrieved knowledge.
- **Scalable and Modular**: Designed for easy integration with various data sources and LLMs.

## Tech Stack
- **Python** (Primary language)
- **LangChain** (For retrieval-augmented generation)
- **FAISS** / **ChromaDB** (Vector database for efficient retrieval)
- **Hugging Face Transformers / OpenAI API** (LLM models)
- **FastAPI** (For API deployment)
- **Streamlit** (Frontend for interactive QnA experience)


## Usage
```sh
# Run the backend API
uvicorn app:main --host 0.0.0.0 --port 8000

# Run the Streamlit UI
streamlit run app.py
```

## How It Works
1. **Query Processing**: The user inputs a question.
2. **Document Retrieval**: The system fetches relevant documents from a knowledge base.
3. **LLM Response Generation**: The LLM generates an answer using retrieved context.
4. **Validation Mechanism**: The response is validated against trusted sources.
5. **Final Response**: The validated response is displayed to the user.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
MIT License. See `LICENSE` for details.

