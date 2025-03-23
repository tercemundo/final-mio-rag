import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv

class SimpleEmbeddings(FakeEmbeddings):
    """Clase simple de embeddings que funciona sin dependencias adicionales."""
    def __init__(self, size=384):
        super().__init__(size=size)
    
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text):
        import hashlib
        import numpy as np
        
        # Genera un hash del texto para crear un embedding determinístico
        hash_object = hashlib.md5(text.encode())
        seed = int(hash_object.hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        # Genera un embedding pseudo-aleatorio pero determinístico
        return np.random.rand(self.size).astype(np.float32)

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
    
    # Cargar PDFs
    uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and not st.session_state.pdfs_processed:
        with st.spinner("Procesando PDFs..."):
            try:
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
                
                if not documents:
                    st.error("No se pudieron cargar documentos.")
                    st.stop()
                
                # Dividir texto en chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                chunks = text_splitter.split_documents(documents)
                
                # Usar embeddings simples que no requieren dependencias adicionales
                embeddings = SimpleEmbeddings()
                
                # Crear vectorstore
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Configurar Groq LLM
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model_name=model_name
                )
                
                # Configurar el mensaje de sistema para el RAG
                prompt_template = """
                Eres un asistente experto que responde preguntas basándose en los documentos proporcionados.
                
                Contexto:
                {context}
                
                Pregunta: {question}
                
                Por favor, responde la pregunta de manera concisa y basándote solo en la información proporcionada.
                """
                
                # Configurar RAG
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
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
                import traceback
                st.code(traceback.format_exc())

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
            try:
                with st.spinner("Pensando..."):
                    response = st.session_state.qa_chain.run(prompt)
                    st.markdown(response)
                
                # Guardar respuesta
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error al generar respuesta: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    if not uploaded_files:
        st.info("Por favor, sube al menos un archivo PDF en la barra lateral para comenzar.")
