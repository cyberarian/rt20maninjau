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
from datetime import datetime
import toml
import chromadb
from st_aggrid import AgGrid, GridOptionsBuilder
import sqlite3

# Update the init_members_db function with migration handling
def init_members_db():
    conn = sqlite3.connect("members.db")
    c = conn.cursor()
    
    # First check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='members'")
    table_exists = c.fetchone() is not None
    
    if not table_exists:
        # Create new table with all columns
        c.execute("""
            CREATE TABLE members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                block_number TEXT NOT NULL
            )
        """)
    else:
        # Check if block_number column exists
        c.execute("PRAGMA table_info(members)")
        columns = [column[1] for column in c.fetchall()]
        
        # Add block_number column if it doesn't exist
        if "block_number" not in columns:
            c.execute("ALTER TABLE members ADD COLUMN block_number TEXT DEFAULT 'Not Specified'")
    
    conn.commit()
    conn.close()

# Initialize the database
init_members_db()

# Initialize SQLite databases
def init_db(db_name, table_name, columns):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns)}
        )
    """)
    conn.commit()
    conn.close()

# Initialize form submissions database
init_db(
    db_name="form_submissions.db",
    table_name="form_submissions",
    columns=[
        "submitted_date TEXT",
        "nama_lengkap TEXT",
        "blok_nomor_rumah TEXT",
        "email TEXT",
        "bulan TEXT",
        "pesan_keterangan TEXT",
        "bukti_pembayaran TEXT"
    ]
)

# Initialize feedback submissions database
init_db(
    db_name="feedback_submissions.db",
    table_name="feedback_submissions",
    columns=[
        "submitted_date TEXT",
        "name TEXT",
        "blok_no TEXT",
        "pesan TEXT"
    ]
)
# Global DataFrame to store form submissions
if 'form_submissions' not in st.session_state:
    st.session_state.form_submissions = pd.DataFrame(columns=[
        "Submitted Date", "Nama Lengkap", "Blok/Nomor Rumah", "Email", "Bulan", "Pesan/Keterangan", "Bukti Pembayaran"
    ])
    
    # Load existing submissions from the CSV file (if it exists)
    if os.path.exists("form_submissions.csv"):
        st.session_state.form_submissions = pd.read_csv("form_submissions.csv")

# Global DataFrame to store feedback submissions
if 'feedback_submissions' not in st.session_state:
    st.session_state.feedback_submissions = pd.DataFrame(columns=[
        "Submitted Date", "Name", "Blok/No", "Pesan"
    ])
    
    # Load existing feedback submissions from the CSV file (if it exists)
    if os.path.exists("feedback_submissions.csv"):
        st.session_state.feedback_submissions = pd.read_csv("feedback_submissions.csv")
        
# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the config.toml file
config = toml.load(".streamlit/config.toml")

# Apply the custom CSS
st.markdown(f"<style>{config['custom_css']['css']}</style>", unsafe_allow_html=True)
# Add custom CSS for blue font in AwesomeTable
st.markdown(
    """
    <style>
    .awesome-table {
        color: blue !important;  /* Set font color to blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

def show_landing_page():
    """Display the landing page with a background image and transparent content container."""
    # Set the background image URL
    background_image_url = "assets/logo3.png"  # Update this path to your image

    # Debug: Display the image to verify the path
    st.image(background_image_url, use_container_width=True)

    # Use custom CSS to set the background image and style the content container
    st.markdown(
        f"""
        <style>
        /* Set the background image for the entire app */
        [data-testid="stAppViewContainer"] {{
            background-image: url("{background_image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed; /* Optional: Fix the background while scrolling */
        }}

        /* Ensure the background image covers the entire viewport */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: transparent;
        }}

        /* Style the transparent content container */
        .landing-container {{
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
            border-radius: 15px; /* Rounded corners */
            padding: 20px; /* Padding inside the container */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            margin: 0 auto; /* Center the container */
            max-width: 800px; /* Limit the width of the container */
        }}

        .landing-container h1, .landing-container h2, .landing-container h3 {{
            text-align: center; /* Center align headings */
        }}

        .landing-container ul {{
            list-style-type: disc; /* Bullet points for lists */
            padding-left: 40px; /* Indent lists */
        }}

        .centered {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Open the transparent container
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)

    # Title and Introduction
    st.title("Selamat Datang di Aplikasi RT20Maninjau")
    st.write("""
    ### Layanan Publik Berbasis Teknologi Modern
    Aplikasi RT20Maninjau merupakan inisiatif dari Pengurus RT 20/RW 09, Cluster Maninjau, Desa Suradita, Kecamatan Cisauk, Kabupaten Tangerang, untuk memberikan layanan publik yang lebih efisien, transparan, dan responsif kepada seluruh warga. 
    
    Dengan memanfaatkan teknologi terkini, kami berkomitmen untuk meningkatkan kualitas pelayanan dan memudahkan akses informasi bagi seluruh warga.
    
    Dirancang untuk menyediakan berbagai layanan publik secara terintegrasi, aplikasi ini memungkinkan warga untuk:
    - Mengakses informasi terkini tentang kegiatan dan program RT 20.
    - Mengajukan pertanyaan atau permohonan bantuan melalui antarmuka chatbot.
    - Mengisi formulir online untuk keperluan administrasi, seperti pembayaran iuran bulanan dan saran/kritik.
    - Mendapatkan panduan dan informasi seputar layanan publik di lingkungan RT 20.
    """)

    st.markdown("### Daftar Menjadi Anggota")
    with st.form("signup_form", clear_on_submit=True):
        name = st.text_input("Nama Lengkap*", key="signup_name")
        block_number = st.text_input("Blok/No. Rumah*", key="signup_block", placeholder="A1/20")
        email = st.text_input("Email*", key="signup_email")
        password = st.text_input("Password*", type="password", key="signup_password")
        submitted = st.form_submit_button("Daftar")
        
        if submitted:
            if not all([name, email, password, block_number]):
                st.error("❌ Mohon lengkapi semua field yang wajib (*)")
            else:
                # Save the member data to the database
                conn = sqlite3.connect("members.db")
                c = conn.cursor()
                try:
                    c.execute("""
                        INSERT INTO members (name, email, password, block_number)
                        VALUES (?, ?, ?, ?)
                    """, (name, email, password, block_number))
                    conn.commit()
                    st.success("✅ Pendaftaran berhasil! Silakan login untuk mengakses formulir.")
                except sqlite3.IntegrityError:
                    st.error("❌ Email sudah terdaftar. Silakan gunakan email lain.")
                finally:
                    conn.close()

    # Centered "Access Admin Panel" button
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    if st.button("Klik disini lebih lanjut", key="access_admin_button"):
        st.session_state['show_admin'] = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the transparent container
    st.markdown('</div>', unsafe_allow_html=True)

def show_login_form(key="login_form"):
    """Display the login form with a unique key."""
    st.markdown("### Login untuk Mengakses Formulir")
    with st.form(key=key, clear_on_submit=True):  # Use the provided key
        email = st.text_input("Email*", key=f"{key}_email")
        password = st.text_input("Password*", type="password", key=f"{key}_password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if not all([email, password]):
                st.error("❌ Mohon lengkapi semua field yang wajib (*)")
            else:
                # Check if the member exists in the database
                conn = sqlite3.connect("members.db")
                c = conn.cursor()
                c.execute("SELECT * FROM members WHERE email = ? AND password = ?", (email, password))
                member = c.fetchone()
                conn.close()
                
                if member:
                    st.session_state['logged_in'] = True
                    st.session_state['member_name'] = member[1]  # Store the member's name
                    st.success(f"✅ Selamat datang, {member[1]}! Anda sekarang dapat mengakses formulir.")
                    
                    # Force a rerun to update the UI
                    st.rerun()
                else:
                    st.error("❌ Email atau password salah. Silakan coba lagi.")
                    
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
        if st.sidebar.checkbox("Are you sure? This will delete all processed documents and submissions."):
            try:
                # Clear cache first
                clear_cache()
                
                # Clear vector store
                if os.path.exists(CHROMA_DB_DIR):
                    shutil.rmtree(CHROMA_DB_DIR)
                    os.makedirs(CHROMA_DB_DIR)
                    st.session_state.uploaded_file_names.clear()
                    st.session_state.vectorstore = None
                
                # Clear form submissions
                if os.path.exists("form_submissions.csv"):
                    os.remove("form_submissions.csv")
                    st.session_state.form_submissions = pd.DataFrame(columns=[
                        "Submitted Date", "Nama Lengkap", "Blok/Nomor Rumah", "Email", "Bulan", "Pesan/Keterangan", "Bukti Pembayaran"
                    ])
                
                # Clear feedback submissions
                if os.path.exists("feedback_submissions.csv"):
                    os.remove("feedback_submissions.csv")
                    st.session_state.feedback_submissions = pd.DataFrame(columns=[
                        "Submitted Date", "Name", "Blok/No", "Pesan"
                    ])
                
                st.sidebar.success("Complete reset successful!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error during reset: {str(e)}")
                logger.error(traceback.format_exc())
                
def show_form_submissions():
    """Display the form submissions DataFrame in the Admin Panel."""
    if st.session_state.admin_authenticated:
        st.sidebar.header("Form Submissions")
        
        # Check if there are submissions in the session state
        if not st.session_state.form_submissions.empty:
            st.sidebar.dataframe(st.session_state.form_submissions)
        # If no submissions in session state, check the CSV file
        elif os.path.exists("form_submissions.csv"):
            df = pd.read_csv("form_submissions.csv")
            st.sidebar.dataframe(df)
        else:
            st.sidebar.info("No form submissions yet.")

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
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        st.image("assets/logo3.png", width=350)
    
    # Create tabs
    tab1, tab4, tab5, tab3, tab2 = st.tabs(["💬 Chat", "📝 Formulir IPL", "📝 Saran/Kritik", "❓ How-to", "ℹ️ About",])    
    
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
            
        # Footer
        st.markdown("---")
        st.markdown("Built by Ketua RT 20 with help from AI :orange_heart:", help="cyberariani@gmail.com")
    with tab2:
        st.write("""
        ### Menghadirkan Layanan publik berbasis AI untuk warga RT 20/RW 09
        
        Selamat datang di Aplikasi RT20Maninjau, yang bertujuan untuk menghadirkan layanan publik di lingkungan RT 20 dengan menggunakan teknologi Artificial Intelligence (AI). 

        Sebagai Pengurus RT, kami berkomitmen untuk meningkatkan kualitas pelayanan yang lebih baik dan lebih cepat dengan mengintegrasikan AI dalam sistem pelayanan di lingkungan kami.

        Aplikasi RT20Maninjau mengintegrasikan teknologi AI untuk:
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
        # Footer
        st.markdown("---")
        st.markdown("Built by Ketua RT 20 with help from AI :orange_heart:", help="cyberariani@gmail.com")
    with tab3:
        st.header("Cara Menggunakan Chatbot RT 20")
        
        st.subheader("📝 Panduan Dasar")
        st.markdown("""
        * Ketik pertanyaan Anda tentang RT 20 di kotak chat
        * Tunggu beberapa saat untuk mendapatkan jawaban
        * Pertanyaan bisa dalam Bahasa Indonesia, Jawa, Sunda, Inggris juga boleh. Bahasa kalbu dan kode belum saya support😅
        """)
        
        st.subheader("🔍 FAQs (Fertanyaan yAng sering ditanyaQans)")
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
        
        # Footer
        st.markdown("---")
        st.markdown("Built by Ketua RT 20 with help from AI :orange_heart:", help="cyberariani@gmail.com")
        
    with tab4:
        # Check if the user is logged in
        if st.session_state.get('logged_in'):
            st.header("📝 Form Iuran Bulanan")
            st.markdown("Silakan lengkapi form berikut untuk pembayaran iuran:")
            
            with st.form("payment_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    nama = st.text_input("Nama Lengkap*", key="nama")
                    blok = st.text_input("Blok/Nomor Rumah*", key="blok", placeholder="A1/20")
                    email = st.text_input("Email*", key="email")
                
                with col2:
                    bulan = st.multiselect(
                        "Pembayaran Bulan*",
                        ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                        "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
                    )
                    pesan = st.text_area("Pesan/Keterangan", placeholder="Tambahan informasi...")
                    bukti = st.file_uploader("Bukti Pembayaran*", type=['pdf','png','jpg','jpeg'])
                
                submitted = st.form_submit_button("Kirim Form")
                
                if submitted:
                    if not all([nama, blok, email, bulan, bukti]):
                        st.error("❌ Mohon lengkapi semua field yang wajib (*)")
                    else:
                        # Save the form data to the database
                        conn = sqlite3.connect("form_submissions.db")
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO form_submissions (
                                submitted_date, nama_lengkap, blok_nomor_rumah, email, bulan, pesan_keterangan, bukti_pembayaran
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            nama,
                            blok,
                            email,
                            ", ".join(bulan),
                            pesan,
                            bukti.name if bukti else None
                        ))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Form berhasil dikirim!")
                        st.info("💡 Data akan tersimpan di database RT 20")
        else:
            st.warning("🔒 Silakan login untuk mengakses formulir ini.")
            show_login_form(key="login_form_tab4")  # Unique key for Tab 4
            
            # Display the form submissions below the form (only for admins)
        if st.session_state.admin_authenticated:
            st.header("Form Submissions")
            conn = sqlite3.connect("form_submissions.db")
            df = pd.read_sql_query("SELECT * FROM form_submissions", conn)
            conn.close()
            
            if not df.empty:
                # Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_default_column(
                    editable=False,  # Disable editing
                    filterable=True,  # Enable filtering
                    sortable=True,    # Enable sorting
                    resizable=True    # Enable column resizing
                )
                # Enable text wrapping for the "pesan_keterangan" column
                gb.configure_column("pesan_keterangan", wrapText=True, autoHeight=True)
                gb.configure_pagination(
                    paginationAutoPageSize=False,  # Disable auto page size
                    paginationPageSize=10          # Set page size to 10
                )
                grid_options = gb.build()
                
                # Display the table
                AgGrid(
                    df,
                    gridOptions=grid_options,
                    height=400,  # Set table height
                    theme="streamlit",  # Use Streamlit theme
                    fit_columns_on_grid_load=True,  # Fit columns to grid width
                    autoSizeColumns=True  # Automatically adjust column widths
                )
        else:
                st.info("Belum ada form yang dikirim.")
    with tab5:
        # Check if the user is logged in
        if st.session_state.get('logged_in'):
            st.header("📝 Saran/Masukan/Kritik")
            st.markdown("Silakan berikan saran, masukan, atau kritik Anda:")
            
            with st.form("feedback_form", clear_on_submit=True):
                name = st.text_input("Nama*", key="feedback_name")
                blok_no = st.text_input("Blok/No*", key="feedback_blok_no", placeholder="A1/20")
                pesan = st.text_area("Pesan*", placeholder="Tulis saran, masukan, atau kritik Anda...")
                
                submitted = st.form_submit_button("Kirim")
                
                if submitted:
                    if not all([name, blok_no, pesan]):
                        st.error("❌ Mohon lengkapi semua field yang wajib (*)")
                    else:
                        # Save the feedback data to the database
                        conn = sqlite3.connect("feedback_submissions.db")
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO feedback_submissions (
                                submitted_date, name, blok_no, pesan
                            ) VALUES (?, ?, ?, ?)
                        """, (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            name,
                            blok_no,
                            pesan
                        ))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Terima kasih! Saran/masukan/kritik Anda telah berhasil dikirim.")
                        st.info("💡 Data akan tersimpan di database RT 20")
        else:
            st.warning("🔒 Silakan login untuk mengakses formulir ini.")
            show_login_form(key="login_form_tab5")  # Unique key for Tab 5
            
            # Display the feedback submissions below the form (only for admins)
        if st.session_state.admin_authenticated:
            st.header("Feedback Submissions")
            conn = sqlite3.connect("feedback_submissions.db")
            df = pd.read_sql_query("SELECT * FROM feedback_submissions", conn)
            conn.close()
            
            if not df.empty:
                # Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_default_column(
                    editable=False,  # Disable editing
                    filterable=True,  # Enable filtering
                    sortable=True,    # Enable sorting
                    resizable=True    # Enable column resizing
                )
                # Enable text wrapping for the "pesan" column
                gb.configure_column("pesan", wrapText=True, autoHeight=True)
                gb.configure_pagination(
                    paginationAutoPageSize=False,  # Disable auto page size
                    paginationPageSize=10          # Set page size to 10
                )
                grid_options = gb.build()
                
                # Display the table
                AgGrid(
                    df,
                    gridOptions=grid_options,
                    height=400,  # Set table height
                    theme="streamlit",  # Use Streamlit theme
                    fit_columns_on_grid_load=True,  # Fit columns to grid width
                    autoSizeColumns=True  # Automatically adjust column widths
                )
            else:
                st.info("Belum ada saran/masukan/kritik yang diterima.")
            
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
    
    # Initialize session state for showing admin panel
    if 'show_admin' not in st.session_state:
        st.session_state['show_admin'] = False

    # Show landing page if not accessing admin panel
    if not st.session_state['show_admin']:
        show_landing_page()
        return
    
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
            Your name: Pak RT.
            Language: Dynamically adapt your responses to match the user's language with formal tone
            Function: Assist the user in finding only relevant information within the provided documents. Your responses must be concise, accurate, and directly address the user's question. Do not include irrelevant details or assumptions.

            Guidelines:
            1. **Relevance**: Base your responses strictly on the content of the provided documents. Do not provide information outside the context of the documents.
            2. **Conciseness**: Keep responses brief and to the point. Only provide detailed explanations if explicitly requested by the user.
            3. **Accuracy**: Ensure all information is accurate and directly sourced from the documents. Do not guess or infer information.
            4. **Structure**:
            - Start with a direct answer to the question.
            - If necessary, provide supporting details in bullet points or numbered lists.
            - End with a summary or conclusion if the question requires it.
            5. **Language**: Use clear and formal language. Adapt your tone to match the user's language (e.g., formal, informal, technical).

            Examples:
            - If the user asks for a name, provide only the name and its context from the documents.
            - If the user asks for a location, provide only the location and its relevance from the documents.
            - If the user asks for a procedure, provide a step-by-step list only if the steps are explicitly mentioned in the documents.

            Do not:
            - Provide opinions or assumptions.
            - Include information not found in the documents.
            - Add unnecessary explanations or details.

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
    
if __name__ == "__main__":
    main()