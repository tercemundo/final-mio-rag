import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
try:
    # Intentar usar HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USE_HF_EMBEDDINGS = True
except ImportError:
    # Si falla, usar embeddings más simples de OpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
    USE_HF_EMBEDDINGS = False
    
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Verificar API key (priorizar la del entorno)
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("No se encontró la API key de Groq en las variables de entorno. Por favor configúrala como GROQ_API_KEY.")
    st.stop()

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
    
    # Mostrar que se está usando la API key del entorno
    st.success("Usando API key de Groq desde variables de entorno")
    
    # Seleccionar modelo
    model_name = st.selectbox(
        "Selecciona modelo de Groq",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )
    
    # Opciones de embeddings
    if not USE_HF_EMBEDDINGS:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.warning("Se utilizará la API de OpenAI para embeddings. Ingresa una API key o configúrala como OPENAI_API_KEY en el entorno.")
            openai_api_key = st.text_input("OpenAI API Key (para embeddings)", type="password")
    
    # Cargar PDFs
    uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and not st.session_state.pdfs_processed:
        # Verificar si se puede proceder
        can_proceed = True
        if not USE_HF_EMBEDDINGS and not openai_api_key:
            st.error("Se necesita una API key de OpenAI para los embeddings.")
            can_proceed = False
        
        if can_proceed:
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
                try:
                    if USE_HF_EMBEDDINGS:
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        st.success("Usando embeddings de HuggingFace")
                    else:
                        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                        st.success("Usando embeddings de OpenAI")
                    
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
                except Exception as e:
                    st.error(f"Error al procesar los documentos: {str(e)}")

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
    if not uploaded_files:
        st.info("Por favor, sube al menos un archivo PDF en la barra lateral para comenzar.")
