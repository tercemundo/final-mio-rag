# final-mio-rag

## Solución sin FAISS y sin dependencias difíciles de instalar

He creado una versión extremadamente simplificada que:

- **Elimina FAISS completamente**: Ya no necesitarás compilar esta biblioteca que está causando problemas.
- **Implementa un retriever manual básico**: Una simple clase `SimpleRetriever` que hace búsqueda de palabras clave.
- **Reduce las dependencias al mínimo**: Solo usa los paquetes esenciales.

## Cómo funciona esta versión:

1. **Procesamiento de PDFs**: Igual que antes, carga y divide los PDFs en fragmentos más pequeños.
2. **Búsqueda simplificada**: Busca coincidencias entre las palabras de la pregunta y los documentos.
3. **Generación de respuestas con contexto**: Envía los documentos relevantes y la pregunta a Groq.

## Ventajas de esta implementación:

- **Mínimas dependencias**: Mucho menos probable que tengas problemas de instalación.
- **Rápida instalación**: No requiere compilación de paquetes complejos como FAISS.
- **Funcionalidad básica completa**: Aún permite cargar PDFs y hacer preguntas sobre ellos.

## Instrucciones de uso:

1. Crea un nuevo archivo `rag.py` con el código proporcionado.
2. Crea un nuevo `requirements.txt` con el contenido del archivo `requirements-ultra-minimal.txt`.
3. Instala las dependencias: 
   ```bash
   pip install -r requirements.txt
