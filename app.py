from library import *
from utils import *

def main():

    load_dotenv()

    st.set_page_config(page_title="CHATBOT", page_icon="👾")

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        
        # Lógica de procesamiento de archivos
        pdf_docs  = ["./Documents\TESTER2.pdf"]

        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks, os.path.basename(pdf_docs[0]).replace('.pdf', ''))
        st.session_state.conversation = get_conversation_chain(vectorstore)

    if "chat_history" not in st.session_state:
        file_path = "./chat_history.pkl"
        # Intenta cargar el historial desde un archivo, si no existe, inicializa como lista vacía
        try:
            with open(file_path, "rb") as f:
                st.session_state.chat_history = pickle.load(f)
        except (FileNotFoundError, EOFError):
            st.session_state.chat_history = []

    # Cargar la imagen
    logo = Image.open('./assets/accenture.png')
    half = 0.1
    logo = logo.resize([int(half * s) for s in logo.size])

    # Crear un contenedor para envolver la imagen
    container = st.container()
    container.width = 100
    container.height = 100

    # Mostrar la imagen en el contenedor
    container.image(
        logo,
        width=100,
    )
    st.header("🗨️ Comienza a preguntar")
    #st.image('./assets/accenture.png', width=100)

    user_question = st.text_input("Realiza una consulta")

    #header_container.markdown('<p style="text-align: right;">Texto en la esquina superior derecha</p>', unsafe_allow_html=True)


    if user_question:
        processing_container = st.empty()
        processing_container.image("./assets/icegif.gif", caption="Procesando...", use_column_width=True , width=200)

        # Procesamiento real del chatbot
        source = handle_userinput(user_question)

        # Ocultar o quitar el gif animado después del procesamiento
        processing_container.empty()

        # Mostrar la respuesta del chatbot
        #st.dataframe(source)
        add_vertical_space(5) 
    
    save_chat_history(st.session_state.chat_history)
    # Llamas a esta función cuando quieras eliminar el historial
    #delete_chat_history()



if __name__ == '__main__':
    main()
