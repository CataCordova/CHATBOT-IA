from library import *

def get_pdf_text(pdf_docs):   #Esta función toma una lista de documentos PDF y extrae el texto de cada página de cada documento.
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):  #Divide el texto en fragmentos más pequeños utilizando RecursiveCharacterTextSplitter. Estos fragmentos se utilizan más adelante para crear vectores de texto.
    text_splitter = RecursiveCharacterTextSplitter( #Para dividir el texto en fragmentos.
        #separator="\n",
        chunk_size=1500,
        #chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text) #Se utiliza el método split_text de la instancia de RecursiveCharacterTextSplitter para dividir el texto en fragmentos. Estos fragmentos se almacenan en la lista chunks y se devuelven al final de la función.
    return chunks #Los fragmentos de texto resultantes se utilizan más adelante para crear vectores de texto.

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
def get_embedding(text):
    return model.encode(text)


pdf_docs = ["documento.pdf", "documento2.pdf"]  # Lista de nombres de archivos PDF

# Obtener el texto de los PDFs
text = get_pdf_text(pdf_docs)

# Dividir el texto en fragmentos
chunks = get_text_chunks(text)

# Generar incrustaciones para los fragmentos
embeddings = get_embedding(chunks)

# Imprimir las incrustaciones para verificar
for embedding in embeddings:
    print(embedding)