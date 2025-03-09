import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import pickle
from transformers import pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from evaluate import load
import logging
import shap
from datetime import datetime
import schedule
import dask.dataframe as dd

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("🚀 Advanced RAG Validation Framework & Conversational RAG")
st.write("Validate RAG-based systems and interact with PDF content using advanced metrics and conversational AI.")

# Initialize validation pipelines
bias_detector = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")
toxicity_detector = pipeline("text-classification", model="unitary/toxic-bert")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Function to preprocess text for bias and toxicity detection
def preprocess_text(text, max_length=512):
    """
    Truncate text to fit within the model's token limit.
    """
    return text[:max_length]

# Function to detect bias
def detect_bias(text):
    """
    Detect bias in the input text using a pre-trained model.
    """
    processed_text = preprocess_text(text)
    bias_result = bias_detector(processed_text)
    return bias_result

# Function to detect toxicity
def detect_toxicity(text):
    """
    Detect toxicity in the input text using a pre-trained model.
    """
    processed_text = preprocess_text(text)
    toxicity_result = toxicity_detector(processed_text)
    return toxicity_result

# Function to calculate accuracy metrics
def calculate_accuracy_metrics(predicted, ground_truth):
    """
    Calculate BLEU, ROUGE, METEOR, and BERTScore for accuracy.
    """
    # BLEU Score
    bleu = sentence_bleu([ground_truth.split()], predicted.split())

    # ROUGE Scores
    rouge_scores = scorer.score(ground_truth, predicted)

    # METEOR Score
    meteor = meteor_score([ground_truth.split()], predicted.split())

    # BERTScore
    P, R, F1 = bert_score([predicted], [ground_truth], lang="en")

    return {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "METEOR": meteor,
        "BERTScore": F1.mean().item()
    }

# Function to check consistency using Hugging Face embeddings
def check_consistency(responses):
    """
    Check consistency across multiple responses using Hugging Face embeddings.
    """
    embeddings_list = embeddings.embed_documents(responses)
    similarity_matrix = cosine_similarity(embeddings_list)
    return similarity_matrix

# Function to evaluate responses using Groq's LLM
def evaluate_with_llm(query, response, groq_api_key):
    """
    Evaluate the quality of a response using Groq's LLM.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")
    prompt = f"""
    Evaluate the following response to the query:
    Query: {query}
    Response: {response}

    Provide feedback on:
    1. Relevance: Is the response relevant to the query?
    2. Correctness: Is the response factually correct?
    3. Coherence: Is the response coherent and well-structured?
    4. Completeness: Does the response fully address the query?

    Return your evaluation in JSON format.
    """
    evaluation = llm.invoke(prompt)
    return evaluation.content

# Function to calculate perplexity
def calculate_perplexity(text):
    """
    Calculate perplexity using Hugging Face's evaluate library.
    """
    perplexity = load("perplexity")
    results = perplexity.compute(predictions=[text], model_id="gpt2")
    return results["perplexities"][0]


# Function to calculate semantic similarity
def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity using Sentence-BERT.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = util.cos_sim(embeddings1, embeddings2)
    return similarity.item()



# Function to detect factual consistency
def check_factual_consistency(predicted, ground_truth):
    """
    Check factual consistency using a pre-trained model.
    """
    fact_checker = pipeline("text-classification", model="facebook/bart-large-mnli")
    input_text = f"claim: {predicted} evidence: {ground_truth}"
    result = fact_checker(input_text)

    if result and isinstance(result, list) and 'label' in result[0]:
        label = result[0]['label']
        score = result[0]['score']
        return label, str(score)  # Ensure score is returned as a string

    return "Error", "0.0"  # Ensure fallback values are strings

# Function to generate SHAP explanations
def generate_shap_explanation(text, llm):
    """
    Generate SHAP explanations for a given text.
    """
    explainer = shap.Explainer(llm)
    shap_values = explainer([text])
    return shap_values

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["RAG Validation", "Conversational RAG"])

if app_mode == "RAG Validation":
    st.header("RAG Validation Framework")
    st.write("Validate the correctness, reliability, and ethical use of RAG-based systems using deterministic approaches.")

    # File uploader for ground truth and predicted responses
    uploaded_file = st.file_uploader("Upload a CSV file with ground truth and predicted responses", type="csv")

    if uploaded_file:
        # Load the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.write(df.head())

        # Ensure the CSV has the required columns
        if "ground_truth" not in df.columns or "predicted" not in df.columns:
            st.error("The CSV file must contain 'ground_truth' and 'predicted' columns.")
        else:
            # Calculate metrics for each row
            accuracy_results = []
            bias_results = []
            toxicity_results = []
            llm_evaluations = []
            perplexity_results = []
            semantic_similarity_results = []
            factual_consistency_results = []

            for index, row in df.iterrows():
                ground_truth = row["ground_truth"]
                predicted = row["predicted"]

                # Accuracy Metrics
                accuracy_metrics = calculate_accuracy_metrics(predicted, ground_truth)
                accuracy_results.append(accuracy_metrics)

                # Bias Detection
                bias_result = detect_bias(predicted)
                bias_results.append(bias_result[0]['score'])

                # Toxicity Detection
                toxicity_result = detect_toxicity(predicted)
                toxicity_results.append(toxicity_result[0]['score'])

                # LLM-Based Evaluation
                llm_evaluation = evaluate_with_llm(row["query"], predicted, os.getenv("GROQ_API_KEY"))
                llm_evaluations.append(llm_evaluation)

                # Perplexity
                perplexity = calculate_perplexity(predicted)
                perplexity_results.append(perplexity)

                # Semantic Similarity
                similarity = calculate_semantic_similarity(predicted, ground_truth)
                semantic_similarity_results.append(similarity)

                actual_consistency = check_factual_consistency(predicted, ground_truth)
                factual_consistency_results.append(actual_consistency)  # Use the correct variable

            # Add results to the DataFrame
            df["BLEU"] = [result["BLEU"] for result in accuracy_results]
            df["ROUGE-1"] = [result["ROUGE-1"] for result in accuracy_results]
            df["ROUGE-L"] = [result["ROUGE-L"] for result in accuracy_results]
            df["METEOR"] = [result["METEOR"] for result in accuracy_results]
            df["BERTScore"] = [result["BERTScore"] for result in accuracy_results]
            df["Bias Confidence"] = bias_results
            df["Toxicity Confidence"] = toxicity_results
            df["LLM Evaluation"] = llm_evaluations
            df["Perplexity"] = perplexity_results
            df["Semantic Similarity"] = semantic_similarity_results
            df["Factual Consistency"] = factual_consistency_results

            # Display results
            st.write("### Validation Results")
            st.write(df)

            # Visualization Dashboard
            st.write("### Visualization Dashboard")

            # Bar Chart for Accuracy Metrics
            st.write("#### Accuracy Metrics")
            accuracy_df = df[["BLEU", "ROUGE-1", "ROUGE-L", "METEOR", "BERTScore"]].mean().reset_index()
            accuracy_df.columns = ["Metric", "Value"]
            fig = px.bar(accuracy_df, x="Metric", y="Value", title="Accuracy Metrics")
            st.plotly_chart(fig)

            # Pie Chart for Bias and Toxicity
            st.write("#### Bias and Toxicity Confidence")
            bias_toxicity_df = pd.DataFrame({
                "Type": ["Bias", "Toxicity"],
                "Confidence": [df["Bias Confidence"].mean(), df["Toxicity Confidence"].mean()]
            })
            fig = px.pie(bias_toxicity_df, values="Confidence", names="Type", title="Bias and Toxicity Confidence")
            st.plotly_chart(fig)

        

            # Perplexity Distribution Histogram
            st.write("#### Perplexity Distribution")
            fig4 = px.histogram(df, x="Perplexity", nbins=20, title="Perplexity Distribution")
            st.plotly_chart(fig4)

            # Semantic Similarity Scatter Plot
            st.write("#### Semantic Similarity Scatter Plot")
            fig5 = px.scatter(df, x="Semantic Similarity", y="BERTScore", title="Semantic Similarity vs BERTScore")
            st.plotly_chart(fig5)

            # Factual Consistency Bar Chart
            st.write("#### Factual Consistency")
            factual_consistency_df = df["Factual Consistency"].value_counts().reset_index()
            factual_consistency_df.columns = ["Consistency", "Count"]
            fig6 = px.bar(factual_consistency_df, x="Consistency", y="Count", title="Factual Consistency")
            st.plotly_chart(fig6)

            # LLM Evaluation Results
            st.write("#### LLM Evaluation Results")
            st.write(df["LLM Evaluation"])

elif app_mode == "Conversational RAG":
    st.header("📚 Conversational RAG with PDF Uploads and Chat History")
    st.write("Upload PDFs and chat with their content. Powered by Groq and LangChain.")

    # Manage session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")  # Use a powerful model
        except Exception as e:
            st.error(f"Invalid API key or Groq service error: {e}")
            st.stop()

        # Session management
        session_id = st.text_input("Session ID", value="default_session")

        if st.button("New Session"):
            session_id = f"session_{len(st.session_state.store) + 1}"
            st.session_state.store[session_id] = ChatMessageHistory()
            st.success(f"New session created: {session_id}")

        # File uploader for PDFs
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        # Process uploaded PDFs
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                documents = []
                for idx, uploaded_file in enumerate(uploaded_files):
                    temp_pdf = f"./temp_{idx}.pdf"  # Create unique filenames
                    with open(temp_pdf, "wb") as file:
                        file.write(uploaded_file.getvalue())

                    # Load PDF content
                    try:
                        loader = PyPDFLoader(temp_pdf)
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {e}")
                        continue

                # Clean up temporary files
                for idx in range(len(uploaded_files)):
                    temp_pdf = f"./temp_{idx}.pdf"
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)

                # Split and create embeddings for documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                splits = text_splitter.split_documents(documents)

                # Use FAISS for vector storage
                faiss_index_path = "faiss_index.pkl"
                if os.path.exists(faiss_index_path):
                    with open(faiss_index_path, "rb") as f:
                        vectorstore = pickle.load(f)
                else:
                    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
                    with open(faiss_index_path, "wb") as f:
                        pickle.dump(vectorstore, f)

                retriever = vectorstore.as_retriever()

            st.success("PDFs processed and embeddings generated!")

            # Contextualization prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question, "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is. "
                "Ensure that the reformulated question is precise and captures "
                "even the smallest details mentioned in the documents."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Question-answering system prompt with strict relevance filter
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use ONLY the following pieces of retrieved context to answer "
                "the question. If the question is not related to the uploaded PDFs, "
                "respond with 'This question is not related to the uploaded documents.' "
                "If you don't know the answer, say that you don't know. "
                "Ensure that you capture even the smallest details from the documents. "
                "Do not provide any information outside the context of the uploaded documents. "
                "Keep the answer concise and to the point."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Function to get session history
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            # Define the conversational RAG chain
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # User question input
            user_input = st.text_input("Your question:")
            if user_input:
                with st.spinner("Generating response..."):
                    session_history = get_session_history(session_id)
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        st.write("### Assistant:")
                        st.write(response['answer'])
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

                st.write("### Chat History:")
                for message in session_history.messages:
                    st.write(f"- {message}")

            # Export chat history
            if st.button("Export Chat History"):
                chat_history_text = "\n".join([str(msg) for msg in session_history.messages])
                st.download_button(
                    label="Download Chat History",
                    data=chat_history_text,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
    else:
        st.warning("Please set the GROQ_API_KEY environment variable.")

# Run the application
if __name__ == "__main__":
    st.write("Ready for conversation with uploaded PDFs!")