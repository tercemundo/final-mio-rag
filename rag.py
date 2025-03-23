import streamlit as st
import os
import tempfile
import numpy as np
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Verificar API key (priorizar la del entorno)
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("No se encontró la API key de Groq en las variables de entorno. Por favor configúrala como GROQ_API_KEY.")
    st.stop()

# Clase simple para almacenamiento y recuperación de texto
class SimpleRetriever:
    def __init__(self):
        self.documents = []
    
    def add_documents(self, documents):
        self.documents.extend(documents)
    
    def get_relevant_documents(self, query, k=3):
        # Búsqueda básica basada en coincidencia de palabras
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            # Contar cuántas palabras de la consulta aparecen en el documento
            word_matches = sum(1 for word in query_words if word in content)
            # Calcular puntuación de similitud simple
            score = word_matches / len(query_words) if query_words else 0
            scored_docs.append((doc, score))
        
        # Ordenar por puntuación y devolver los k mejores
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

# Configurar página de Streamlit
st.set_page_config(page_title="Chat con tus PDFs usando Groq", layout="wide")
st.title("Chat con tus PDFs usando Groq")

# Inicializar variables de estado en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

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
                
                # Crear retriever simple
                retriever = SimpleRetriever()
                retriever.add_documents(chunks)
                st.session_state.retriever = retriever
                
                # Configurar Groq LLM
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model_name=model_name
                )
                st.session_state.llm = llm
                
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

# Función para generar respuesta
def generate_response(query):
    # Obtener documentos relevantes
    docs = st.session_state.retriever.get_relevant_documents(query, k=5)
    
    # Crear contexto a partir de los documentos
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Crear mensaje con el contexto y la pregunta
    system_message = """
    Eres un asistente experto que responde preguntas basándose en los documentos proporcionados.
    Responde de manera concisa y basándote solo en la información proporcionada.
    Si la información necesaria no está en los documentos, indícalo claramente.
    """
    
    user_message = f"""
    Contexto de los documentos:
    {context}
    
    Mi pregunta es: {query}
    """
    
    # Generar respuesta
    response = st.session_state.llm.invoke(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": user_message}]
    )
    
    return response.content

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
                    response = generate_response(prompt)
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
