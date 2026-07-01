import os
import streamlit as st
import tempfile
import time
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFLoader  
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()

# API Keys setup
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Prompt Template for QA
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Vector Embedding and Summarization Function
def vector_embedding(file_path):
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    st.session_state.loader = PyPDFLoader(file_path)
    st.session_state.docs = st.session_state.loader.load()
    
    # --- New Feature: Generate Summary ---
    with st.spinner("Generating document summary..."):
        # Combine the first few pages to avoid hitting token limits for huge PDFs
        combined_text = " ".join([doc.page_content for doc in st.session_state.docs[:5]]) 
        summary_prompt = f"Provide a concise summary of the following text with 10% length of the original document highlighting the main points:\n\n{combined_text}"
        
        # Invoke LLM directly for the summary
        summary_response = llm.invoke(summary_prompt)
        st.session_state.summary = summary_response.content
    # -------------------------------------

    # Text Splitting
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
    # FAISS Vector Store creation
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Handle file processing
if uploaded_file is not None:
    if st.button("Process Document and Create Vector Store"):
        with st.spinner("Processing PDF... Please wait."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            vector_embedding(tmp_file_path)
            os.remove(tmp_file_path)
            st.success("Vector Store DB is ready!")

# Persistent UI container to keep the summary visible after page reruns
if "summary" in st.session_state:
    with st.expander("📝 View Document Summary", expanded=True):
        st.write(st.session_state.summary)

# Ask questions block
prompt1 = st.text_input("What do you want to ask from the documents?")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please upload a PDF and click 'Process Document' button first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        
        st.subheader("Response:")
        st.write(response['answer'])

        # Document Similarity Search Expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Source Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("-------------------")
