import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar página de Streamlit
st.set_page_config(page_title="Chat con tus PDFs usando Groq", layout="wide")
st.title("Chat con tus PDFs usando Groq")

# Inicializar variables de estado en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    
    # API Key de Groq
    groq_api_key = st.text_input("Groq API Key", placeholder="Ingresa tu API key de Groq", type="password")
    if not groq_api_key and os.environ.get("GROQ_API_KEY"):
        groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Seleccionar modelo
    model_name = st.selectbox(
        "Selecciona modelo de Groq",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )
    
    # Cargar PDFs
    uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and not st.session_state.pdfs_processed and groq_api_key:
        with st.spinner("Procesando PDFs..."):
            # Guardar archivos en ubicaciones temporales
            temp_files = []
            for file in uploaded_files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_file.write(file.read())
                temp_files.append(temp_file.name)
                temp_file.close()
            
            # Cargar y procesar PDFs
            documents = []
            for path in temp_files:
                try:
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                except Exception as e:
                    st.error(f"Error al cargar {path}: {e}")
            
            # Dividir texto en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            # Configurar embeddings y vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Configurar Groq LLM
            llm = ChatGroq(
                api_key=groq_api_key,
                model_name=model_name
            )
            
            # Configurar RAG
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            
            # Limpiar archivos temporales
            for path in temp_files:
                try:
                    os.unlink(path)
                except:
                    pass
            
            st.session_state.pdfs_processed = True
            st.success(f"Se procesaron {len(chunks)} fragmentos de texto de {len(uploaded_files)} PDFs")

# Área principal para el chat
if st.session_state.pdfs_processed:
    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input para nuevas preguntas
    if prompt := st.chat_input("Pregunta sobre tus PDFs"):
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.qa_chain.run(prompt)
                st.markdown(response)
        
        # Guardar respuesta
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    if not groq_api_key:
        st.warning("Por favor, ingresa tu API key de Groq en la barra lateral.")
    
    if not uploaded_files:
        st.info("Por favor, sube al menos un archivo PDF en la barra lateral para comenzar.")
