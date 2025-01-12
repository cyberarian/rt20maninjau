#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
import fitz  # PyMuPDF
import pandas as pd
import logging
import traceback
import gc
import sys
import shutil
from stqdm import stqdm
from contextlib import contextmanager
from typing import List
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.globals import set_verbose
from dotenv import load_dotenv
from streamlit.runtime.caching import cache_data, cache_resource
import toml
import chromadb
#import sqlite3

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the config.toml file
config = toml.load(".streamlit/config.toml")

# Apply the custom CSS
st.markdown(f"<style>{config['custom_css']['css']}</style>", unsafe_allow_html=True)

# Load the admin password from the .env file
admin_password = os.getenv('ADMIN_PASSWORD')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory management context
@contextmanager
def memory_track():
    try:
        gc.collect()
        yield
    finally:
        gc.collect()

def setup_admin_sidebar():
    """Setup admin authentication and controls in sidebar"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    with st.sidebar:
        st.title("Admin Panel")

        # Admin authentication
        if not st.session_state.admin_authenticated:
            input_password = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                # Use the admin password from the .env file
                if input_password == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated!")
                    st.rerun()
                else:
                    st.error("Incorrect password")
        else:
            st.write("✅ Admin authenticated")
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

            # Show admin controls only when authenticated
            st.divider()
            show_admin_controls()

def show_admin_controls():
    """Display admin controls when authenticated"""
    st.sidebar.header("Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Process documents button
    if uploaded_files:
        if st.sidebar.button("Process Documents", key="process_docs_button"):
            process_uploaded_files(uploaded_files)
    
    # Show currently processed files
    if st.session_state.uploaded_file_names:
        st.sidebar.write("Processed Documents:")
        for filename in st.session_state.uploaded_file_names:
            st.sidebar.write(f"- {filename}")
    
    # Reset system
    st.sidebar.divider()
    st.sidebar.header("System Reset")
    if st.sidebar.button("Reset Everything", key="reset_everything_button"):
        if st.sidebar.checkbox("Are you sure? This will delete all processed documents."):
            try:
                # Clear cache first
                clear_cache()
                
                # Clear vector store
                if os.path.exists(CHROMA_DB_DIR):
                    shutil.rmtree(CHROMA_DB_DIR)
                    os.makedirs(CHROMA_DB_DIR)
                    st.session_state.uploaded_file_names.clear()
                    st.session_state.vectorstore = None
                
                st.sidebar.success("Complete reset successful!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error during reset: {str(e)}")
                logger.error(traceback.format_exc())

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file"""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def get_document_text(file) -> str:
    """Get text content from a file based on its type"""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return file.getvalue().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file.type}")

def process_uploaded_files(uploaded_files: List):
    """Process uploaded files and update the vector store"""
    try:
        # Initialize text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
        
        vectorstore = st.session_state.vectorstore
        
        # Process each file
        with st.spinner('Processing documents...'):
            for file in stqdm(uploaded_files):
                if file.name not in st.session_state.uploaded_file_names:
                    # Extract text based on file type
                    text = get_document_text(file)
                    
                    # Split text into chunks
                    chunks = text_splitter.create_documents([text])
                    
                    # Add metadata to chunks
                    for chunk in chunks:
                        chunk.metadata = {
                            "source": file.name,
                            "chunk_size": len(chunk.page_content)
                        }
                    
                    # Add chunks to vector store
                    vectorstore.add_documents(chunks)
                    
                    # Update processed files list
                    st.session_state.uploaded_file_names.add(file.name)
            
            # No need to call persist() as ChromaDB now handles this automatically
            
        st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents!")
        
    except Exception as e:
        st.sidebar.error(f"Error processing files: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def clear_cache():
    """Clear all cached data"""
    cache_data.clear()
    cache_resource.clear()
    
def show_chat_interface(llm, prompt):
    """Display the main chat interface"""
    # Add logo
    col1, col2, col3 = st.columns([1,9,1])
    with col2:
        st.image("assets/logo4.png", width=350)
    
    # Create tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "ℹ️ About", "❓ How-to"])
    
    with tab1:
        # Add a greeting message
        if not st.session_state.uploaded_file_names:
            st.info("👋 Selamat datang warga RT 20/RW 09, Desa Suradita, Cisauk, Kabupaten Tangerang")
        else:
            st.info("👋 Wilujeng sumping! Punten naroskeun naon waé ngeunaan dokumén anu parantos diunggah.")
        
        # Initialize chat history in session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Create a form for the chat input
        with st.form(key='chat_form'):
            prompt1 = st.text_input("Enter your question about the documents", key='question_input')
            submit_button = st.form_submit_button("Submit Question")
            
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.write("Question:", q)
            st.write("Answer:", a)
            st.divider()
        
        if submit_button and prompt1:  # Only process if there's a question and the button is clicked
            try:
                with memory_track():
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = initialize_or_load_vectorstore()
                    
                    vectorstore = st.session_state.vectorstore
                    if len(vectorstore.get()['ids']) > 0:
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retriever = vectorstore.as_retriever()
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        
                        with st.spinner('Searching through documents...'):
                            start = time.process_time()
                            response = retrieval_chain.invoke({'input': prompt1})
                            elapsed_time = time.process_time() - start
                            
                            # Add the new Q&A to the chat history
                            st.session_state.chat_history.append((prompt1, response['answer']))
                            
                            # Display the latest response
                            st.write("Latest Response:")
                            st.write(response['answer'])
                            st.write(f"Response time: {elapsed_time:.2f} seconds")
                            
                            # Clear the input box by rerunning the app
                            st.rerun()
                    else:
                        st.warning("No documents found in the database. Please ask an admin to upload some documents.")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Add a clear chat history button
        if st.session_state.chat_history and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    with tab2:
        st.write("""
        
        ### Menghadirkan Layanan publik berbasis AI untuk warga RT 20/RW 09
        
        Selamat datang di aplikasi "rt20maninjau", yang bertujuan untuk menghadirkan layanan publik di lingkungan RT 20 dengan menggunakan teknologi Artificial Intelligence (AI). 

        Sebagai Pengurus RT, kami berkomitmen untuk meningkatkan kualitas pelayanan yang lebih baik dan lebih cepat dengan mengintegrasikan AI dalam sistem pelayanan di lingkungan kami.

        Aplikasi "rt20maninjau" mengintegrasikan teknologi AI untuk:
        * Layanan personal
        * Informasi real-time
        * Memudahkan komunikasi RT-warga
                
        #### 🎯 Visi & Misi
        Mewujudkan pelayanan warga yang modern, efisien, dan responsif.
        
        #### 🏡 Lokasi
        RT 20/RW 09, Cluster Maninjau,
        Griya Suradita Indah (GSI),
        Desa Suradita, Kecamatan Cisauk,
        Kabupaten Tangerang, 15843, Provinsi BANTEN
        """)
        
    with tab3:
        st.header("Cara Menggunakan Chatbot RT 20")
        
        st.subheader("📝 Panduan Dasar")
        st.markdown("""
        * Ketik pertanyaan Anda tentang RT 20 di kotak chat
        * Tunggu beberapa saat untuk mendapatkan jawaban
        * Pertanyaan bisa dalam Bahasa Indonesia, Jawa, Sunda, Inggris juga boleh. Bahasa kalbu dan kode belum saya support😅
        """)
        
        st.subheader("🔍 Contoh Pertanyaan")
        st.markdown("""
        * "Siapa saja pengurus RT 20?"
        * "Apa saja fasilitas yang tersedia di RT 20?"
        * "Saya mau pindah domisili, bagaimana caranya?"
        * "Saya warga baru, apa yang harus saya lakukan?"
        * "Listrik saya bermasalah, mati lampu"
        * "Saya butuh kontak satpam"
        * "Mesin cuci saya rusak, punya kontak service elektronik gak?"
        * "Sampah saya kok belum diangkat ya?"
        * "WC saya mampet, penuh"
        * "dll."
        """)
        
        st.subheader("⚠️ Penting")
        st.info("""
        * Aplikasi ini tidak merekam percakapan
        * Chatbot hanya menjawab pertanyaan seputar RT 20 dan lingkungan sekitar. Tidak menerima curcol dan gosip🤐
        * Untuk informasi lebih lanjut, silakan hubungi pengurus RT
        """)
        
def initialize_or_load_vectorstore():
    """Initialize or load the vector store for document embeddings"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize or load the existing Chroma database
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
def main():
    # Disable ChromaDB telemetry
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    
    set_verbose(True)
    load_dotenv()
    
    # Load and validate API keys
    groq_api_key = os.getenv('GROQ_API_KEY')
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not groq_api_key or not google_api_key:
        st.error("Missing API keys. Please check your .env file.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Create ChromaDB directory
    global CHROMA_DB_DIR
    CHROMA_DB_DIR = "chroma_db"
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    # Initialize session state
    if 'uploaded_file_names' not in st.session_state:
        st.session_state.uploaded_file_names = set()
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Initialize LLM and prompt template
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        prompt = ChatPromptTemplate.from_template("""
           Your role: Your name is Pak RT.
            Language: Dynamically adapt your responses to match the user's language with formal tone
            Function: Assist the user in finding relevant information within provided documents, including names, titles, locations, history, tables, images, and other relevant texts. Keep responses brief and accurate; provide detailed explanations only when specifically requested.
            Greetings: respond to the greetings accordingly.
            
            Guidelines:
            1. Format responses as follows:
            - Use bullet points for steps, procedures, or stages
            - Present sequential information in numbered lists
            - Break down complex answers into clear bullet points
            
            2. Response Structure:
            - Start with direct answer
            - List steps/procedures with bullets
            - End with relevant context if needed
                                
            3. Content Rules:
            - Base your responses strictly on the document's content and context
            - Provide answers that directly address the user's question only
            - Do not respond user's questions with irrelevant, misleading, or incomplete information
            - Present table data in a clear and logical format for easy understanding
            - Strive for accuracy and relevance in all responses
            
            lang:id-ID
            
            Context:
            {context}
            
            Question: {input}
            """)
            
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

    # Setup sidebar with admin controls
    setup_admin_sidebar()
    
    # Show main chat interface
    show_chat_interface(llm, prompt)

# Footer
    st.markdown("---")
    st.markdown("Built by Ketua RT 20 with help from AI :orange_heart:", help="cyberariani@gmail.com")
if __name__ == "__main__":
    main()