from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_groq import ChatGroq
from langchain_groq import GroqEmbeddings  # Cambiado a GroqEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Importaciones front-end
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno si están en un archivo .env
load_dotenv()

# ---------------------------------------------------------------------
# ---------------- BACK-END: CLASE QUE IMPLEMENTA RAG -----------------
# ---------------------------------------------------------------------
class ChatPDF:
    def __init__(self):
        # Verificar si existe la API key de Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("No se encontró la API key de Groq. Por favor configúrela como variable de entorno.")
            st.stop()
            
        # Inicializar el modelo de Groq
        self.model = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.1
        )
        
        # Inicializar el modelo de embeddings usando Groq
        self.embeddings = GroqEmbeddings(
            model="text-embedding-groq-model",  # Cambia esto al modelo de embeddings de Groq que corresponda
            groq_api_key=api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        self.prompt = self.crear_prompt()
        self.vector_db = None
        self.retriever = None
        self.chain = None

    def crear_prompt(self):
        """Prompt que se presentará al modelo (LLM) para la generación de respuestas"""
        return PromptTemplate.from_template(
            """
            Eres un asistente para tareas de respuesta a preguntas. Utiliza los siguientes elementos de contexto para responder a la pregunta.
            Si no conoces la respuesta, simplemente di que no lo sabes. Utiliza un máximo de tres frases y sé conciso en tu respuesta.
            
            Question: {question}
            Context: {context}
            Answer:
            """
        )

    def cargar_pdf(self, ruta_pdf: str):
        """Carga el PDF y lo sub-divide en 'chunks'"""
        doc = PyPDFLoader(file_path=ruta_pdf).load()
        chunks = self.text_splitter.split_documents(doc)
        return filter_complex_metadata(chunks)

    def crear_vector_db(self, chunks):
        """Crea la base de datos vectorial a partir de los 'chunks' usando FAISS"""
        self.vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        # Opcionalmente guardar en disco
        self.vector_db.save_local("faiss_index")

    def crear_retriever(self):
        """Crea el 'retriever' que se encargará de buscar la información requerida en el PDF"""
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

    def crear_chain(self):
        """Crea la cadena para interconectar el retriever con la pregunta, el prompt, el modelo y
           la respuesta generada por el modelo"""
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ingerir(self, ruta_pdf: str):
        """Ejecuta secuencialmente todos los métodos anteriores."""
        chunks = self.cargar_pdf(ruta_pdf)
        self.crear_vector_db(chunks)
        self.crear_retriever()
        self.crear_chain()

    def preguntar(self, query: str):
        """Recibe una pregunta del usuario y ejecuta la cadena para generar una respuesta."""
        if self.chain is None:
            return "Debes cargar primero un archivo PDF"
        return self.chain.invoke(query)

    def limpiar(self):
        """Limpia la base de datos vectorial, el retriever y la cadena para una nueva consulta."""
        self.vector_db = None
        self.retriever = None
        self.chain = None

    def cargar_vector_db_si_existe(self):
        """Carga la base de datos vectorial desde el disco si existe"""
        if os.path.exists("faiss_index"):
            self.vector_db = FAISS.load_local("faiss_index", self.embeddings)
            return True
        return False


# ---------------------------------------------------------------------
# ---------------- FRONT-END: APLICATIVO WEB EN STREAMLIT -------------
# ---------------------------------------------------------------------

# Fijar título de la aplicación (aparecerá en la parte superior de la ventana)
st.title("Chatear con PDF usando Groq")

# Función para mostrar los mensajes intercambiados entre la IA y el usuario
def mostrar_mensajes():
    st.subheader("Chat")

    # Mostrar mensajes en formato de Chat
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        with st.chat_message("user" if is_user else "assistant"):
            st.write(msg)

    # Limpiar "spinner" pues el modelo ya generó la respuesta
    st.session_state["thinking_spinner"] = st.empty()

def procesar_entrada():
    # Si hay alguna entrada ingresada y si esta contiene texto (y no sólo espacios en blanco)
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # Presentar texto al modelo y mostrar ícono indicando al usuario que se está procesando la info
        with st.session_state["thinking_spinner"], st.spinner(f"Procesando"):
            agent_text = st.session_state["assistant"].preguntar(user_text)

        # Añadir mensajes al historial
        st.session_state["messages"].append((user_text, True))  # True: el mensaje es del usuario
        st.session_state["messages"].append((agent_text, False))  # False: el mensaje es del asistente

        # Limpiar la entrada del usuario después de procesarla
        st.session_state["user_input"] = ""

def leer_almacenar_pdf():
    # Limpiar asistente, mensajes intercambiados y entrada del usuario
    st.session_state["assistant"].limpiar()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Leer y almacenar PDF en archivo temporal (para poder acceder desde Langchain)
    pdf_file = st.session_state["file_uploader"]
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, 'temp.pdf')
    bytes_data = pdf_file.getvalue()
    with open(temp_filepath, "wb") as f:
        f.write(bytes_data)

    # Ingerir el archivo
    with st.session_state["ingestion_spinner"], st.spinner(f"Ingiriendo archivo PDF"):
        st.session_state["assistant"].ingerir(temp_filepath)

def configurar_api_key():
    """Permite al usuario ingresar la API key de Groq si no está configurada"""
    api_key = st.text_input("Introduce tu GROQ_API_KEY:", type="password", key="groq_api_key_input")
    if st.button("Guardar API Key"):
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API Key guardada para esta sesión!")
        st.rerun()

def page():
    """Función para mostrar el aplicativo al usuario"""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        
    # Verificar si la API key está configurada
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("No se ha configurado la API key de Groq")
        configurar_api_key()
        return
        
    # Inicializar el asistente si aún no existe o si la API key ha cambiado
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()

    st.subheader("Cargue un documento PDF")
    st.file_uploader(
        "Cargar archivo",
        type=["pdf"],
        key="file_uploader",
        on_change=leer_almacenar_pdf,
    )

    # Limpiar "spinner" de ingestión
    st.session_state["ingestion_spinner"] = st.empty()

    mostrar_mensajes()
    st.text_input("Mensaje", key="user_input", on_change=procesar_entrada)


if __name__ == "__main__":
    page()
