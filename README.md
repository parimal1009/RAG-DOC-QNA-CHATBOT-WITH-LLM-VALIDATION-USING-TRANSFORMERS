# RAG QnA Bot & LLM Validation

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) QnA Bot** alongside an **LLM Validation** system. The primary goal is to enhance response accuracy and reliability by integrating an information retrieval system with an LLM, ensuring factually correct and contextually relevant answers.

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
This project is licensed under the **MIT License**. See `LICENSE` for details.

---

# RAG Model Evaluation Dashboard

## Overview
This project is a **Streamlit-based web application** for evaluating Retrieval-Augmented Generation (RAG) models using multiple performance metrics. The app provides a user-friendly interface to analyze bias, toxicity, accuracy, factual consistency, semantic similarity, and LLM-based evaluations.

## Features
- **Upload CSV File**: Users can upload a dataset with `prompt`, `retrieved context`, and `generated response` columns.
- **Bias & Toxicity Analysis**: Detects bias and toxicity in generated text.
- **Accuracy Metrics**: Computes **BLEU, ROUGE, METEOR, and BERTScore** to evaluate responses.
- **Factual Consistency**: Assesses factual alignment between retrieved context and generated response.
- **Semantic Similarity**: Measures how closely the generated response aligns with the expected response.
- **LLM-based Evaluation**: Uses **Groq's LLM API** for additional insights.
- **Visualizations**: Displays performance metrics with interactive charts.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/RAG-Evaluation-Dashboard.git
cd RAG-Evaluation-Dashboard

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```
1. Upload a CSV file containing prompts, retrieved contexts, and generated responses.
2. The system computes various evaluation metrics automatically.
3. View detailed analytics and download results.

## Evaluation Metrics
| Metric                 | Description |
|------------------------|-------------|
| **Bias & Toxicity**    | Detects harmful or biased text in responses |
| **BLEU, ROUGE, METEOR** | Measures textual accuracy |
| **BERTScore**         | Evaluates semantic similarity with reference responses |
| **Factual Consistency** | Assesses alignment with retrieved context |
| **LLM-based Evaluation** | Provides additional scoring based on LLM responses |

## Configuration
- Modify API keys in `.env` if needed.
- Customize evaluation thresholds in `config.py`.

## Contributing
Pull requests are welcome! Please open an issue for discussion before making major changes.
