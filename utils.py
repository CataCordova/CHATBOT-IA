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

modelPath = 'paraphrase-multilingual-mpnet-base-v2' 


def get_vectorstore(text_chunks, store_name): #Crea o carga un "vectorstore" que almacena vectores de texto utilizando la biblioteca FAISS. Estos vectores se utilizan para recuperar información relevante más adelante.
    #Estos vectores representan la "similitud semántica" entre diferentes palabras o textos.
    if os.path.exists(f"./vectorstore/{store_name}.pkl"):   #Verifica si ya existe un archivo de vectorstore en la carpeta "vectorstore". Para ello, utiliza la función os.path.exists() de Python.
        with open(f"./vectorstore/{store_name}.pkl", "rb") as f:
            vectorstore = pickle.load(f) #Si el archivo existe, carga el vectorstore desde el archivo utilizando la función pickle.load().
    else: #Si el archivo no existe, se genera un nuevo vectorstore utilizando las funciones FAISS.from_texts() y OpenAIEmbeddings(). Luego, se guarda en el archivo de vectorstore utilizando la función pickle.dump().
        embeddings = HuggingFaceEmbeddings(model_name=modelPath)
        #embeddings = OllamaEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(f"./vectorstore/{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f) 
    return vectorstore


#model = "tiiuae/falcon-40b"
#tokenizer = AutoTokenizer.from_pretrained(model)
'''
pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
'''
def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    llm = Ollama(model="llama2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=vectorstore.as_retriever(k=5),
        memory=memory,
        return_source_documents=True, 
        return_generated_question=True, 
        # condense_question_prompt=prm.PROMPT_SEMANTIC_SEARCH_CONDENCE_QUESTION,
        #combine_docs_chain_kwargs={'prompt': prm.PROMPT_SEMANTIC_SEARCH},
        # combine_docs_chain_kwargs={'prompt': prm.PROMPT_RERANK},
        verbose=True
    )
    return conversation_chain


def handle_userinput(user_question):    #Toma la pregunta del usuario, realiza una conversación utilizando el modelo y el vectorstore, y devuelve la respuesta generada junto con información adicional.

    response = st.session_state.conversation({"question": user_question})
    df_source = pd.DataFrame()


    for i in response['source_documents']:        
        contenido = i.page_content

        df_source_aux = pd.DataFrame({'Contenido':[contenido]})
        df_source = pd.concat([df_source, df_source_aux]).reset_index(drop=True)

    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"🧑‍🚀 **You:** {message.content}")
        else:
            st.write(f"🤖 **MinerBot:** {message.content}")

    st.write('Respuesta regenerada ', response['generated_question'])

    return df_source

def save_chat_history(chat_history):
    # Guarda el historial en un archivo
    with open(f"./chat_history.pkl", "wb") as f:
        pickle.dump(chat_history, f)

def delete_chat_history():
    file_path = "chat_history.pkl"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Historial eliminado exitosamente.")
    else:
        print("El historial no existe.")
