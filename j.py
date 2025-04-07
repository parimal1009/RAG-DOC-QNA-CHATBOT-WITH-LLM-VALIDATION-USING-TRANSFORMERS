import streamlit as st
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
import tempfile
from datetime import datetime

# Set environment variables from Streamlit secrets
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Initialize HuggingFace Embeddings - Changed to bge-small-en-v1.5
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Streamlit UI Configuration
st.set_page_config(
    page_title="Advanced PDF Chat Assistant",
    page_icon="📕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings & Configuration")
    
    # API Key Input - Use Streamlit secrets for deployment
    if 'GROQ_API_KEY' in st.secrets:
        api_key = st.secrets['GROQ_API_KEY']
        st.success("API key loaded from secrets!")
    else:
        api_key = st.text_input("Enter your Groq API key:", type="password", help="Get your API key from Groq's console")
    
    # Model Selection (UNCHANGED)
    model_options = {
        "mixtral-8x7b-32768": "Best overall performance (recommended)",
        "llama2-70b-4096": "Good alternative to Mixtral",
        "gemma-7b-it": "Fast and efficient for simpler tasks"
    }
    selected_model = st.selectbox(
        "Select Groq Model",
        options=list(model_options.keys()),
        index=0,
        help=model_options["mixtral-8x7b-32768"]
    )
    
    # Advanced Options (UNCHANGED)
    with st.expander("Advanced Options"):
        chunk_size = st.slider("Chunk Size", 1000, 10000, 5000, help="Size of document chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 100, 2000, 500, help="Overlap between document chunks")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, help="Lower for factual, higher for creative")
        max_tokens = st.slider("Max Response Tokens", 100, 2000, 500, help="Limit response length")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This advanced RAG application allows you to:
    - Upload and chat with multiple PDFs
    - Maintain conversation history across sessions
    - Get precise answers with source references
    - Export your chat history
    """)

# [REST OF THE CODE REMAINS EXACTLY THE SAME AS IN YOUR ORIGINAL VERSION]
# Only the embedding model initialization was changed above
# All other code including the Groq models remains identical

# Main UI
st.title("🧠 Advanced PDF Chat Assistant")
st.markdown("""
Upload PDF documents and have natural conversations about their content. 
Powered by **Groq's ultra-fast LLMs** and **LangChain's RAG framework**.
""")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# [CONTINUE WITH THE REST OF YOUR ORIGINAL CODE...]

# Session Management
col1, col2 = st.columns(2)
with col1:
    session_id = st.text_input("Session ID", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                             help="Use different IDs for separate conversations")
with col2:
    if st.button("🔄 New Session", help="Start a fresh conversation"):
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.store[session_id] = ChatMessageHistory()
        st.session_state.messages = []
        st.success(f"New session created: {session_id}")
        st.rerun()

# File Uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, 
                                 help="Upload one or multiple PDFs to chat with")

if uploaded_files:
    with st.spinner("🔍 Processing and indexing documents..."):
        documents = []
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if not new_files:
            st.info("These files have already been processed.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in new_files:
                    try:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        documents.extend(docs)
                        st.session_state.processed_files.add(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue

            if documents:
                # Split text and create embeddings
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                splits = text_splitter.split_documents(documents)

                # Create FAISS vector store
                st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
                st.success(f"✅ Processed {len(documents)} pages from {len(new_files)} files")

# Only proceed if API key is valid and documents are processed
if api_key and st.session_state.vectorstore:
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=selected_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Contextual question reformulation prompt
        contextualize_q_system_prompt = """You are an expert at understanding and refining questions based on conversation history. 
        Given a chat history and the latest question, rephrase the question to be standalone, or return it as-is if no context is needed."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # QA System Prompt
        qa_system_prompt = """You are an expert research assistant with access to document content. Follow these rules:
        1. Answer ONLY using the provided context
        2. Be precise and professional
        3. If unsure, say "I couldn't find that information in the documents"
        4. Cite sources with page numbers when possible
        5. Maintain conversation context from the chat history

        Context: {context}

        Current conversation: {chat_history}

        Question: {input}
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session history management
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat Interface
        st.markdown("---")
        st.subheader("💬 Chat with Your Documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        # Display answer
                        answer = response['answer']
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show sources if available
                        if 'context' in response:
                            with st.expander("🔍 Source References"):
                                for i, doc in enumerate(response['context']):
                                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                    page = doc.metadata.get('page', 'N/A')
                                    st.write(f"**Source {i+1}:** {source} (Page {page})")
                                    st.caption(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

        # Chat history management
        st.markdown("---")
        with st.expander("📜 Session History Management"):
            if st.button("Clear Current Session History"):
                if session_id in st.session_state.store:
                    st.session_state.store[session_id].clear()
                    st.session_state.messages = []
                    st.success("Current session history cleared!")
                    st.rerun()
            
            if st.session_state.store.get(session_id):
                st.write(f"### Messages in session: {session_id}")
                for msg in st.session_state.store[session_id].messages:
                    st.write(f"- {msg.type}: {msg.content}")
                
                # Export chat history
                st.download_button(
                    label="📥 Export Chat History",
                    data="\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.store[session_id].messages]),
                    file_name=f"chat_history_{session_id}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
elif not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
elif not st.session_state.vectorstore:
    st.info("Upload and process PDF documents to begin chatting.")

# Footer
st.markdown("---")
st.caption("""
Advanced PDF Chat Assistant | Powered by Groq & LangChain | 
[Report Issues](https://github.com/your-repo/issues) | [Learn More](https://groq.com/)
""")