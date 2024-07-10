import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
# from openai import OpenAI

# Create temporary and vector store directories
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_store")

if not TMP_DIR.exists():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

if not LOCAL_VECTOR_STORE_DIR.exists():
    LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def streamlit_ui():
    st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

    # Custom CSS to style the sidebar
    st.markdown("""
    <style>
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #f8f9fa !important; /* Light background color */
        padding: 20px;
    }
    .css-1d391kg h2 {
        color: #1f77b4; /* Blue text color */
        font-size: 24px; /* Larger font size */
        font-weight: bold;
    }
    .css-1d391kg .stRadio > label {
        font-size: 20px; /* Larger font size for radio options */
        color: #343a40; /* Dark text color */
    }

    /* Chat message styling */
    .chat-box {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .chat-user {
        font-weight: bold;
        color: #1f77b4;
    }

    .chat-ai {
        font-weight: bold;
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio('Select a page:', ['Home', 'Chat With Document'], index=0)

    if choice == 'Home':
        st.title("Welcome to Document Chatbot 📄")
        st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)
        st.write("You can chat with a document by clicking on 'Chat With Document' in the navigation menu.")
    elif choice == 'Chat With Document':
        st.title('NdriAI - Chat With Document')
        st.write('Upload a document that you want to chat with:')

        source_docs = st.file_uploader(label='Upload a document', type=['pdf'], accept_multiple_files=True)

        if not source_docs:
            st.warning("Please upload a document")
        else:
            query = st.text_input("Enter your query here:")
            if st.button('Send'):
                if query:
                    RAG(source_docs, query)
                else:
                    st.warning("Please enter a query")

        # Display the chat history
            # Display the chat history
        for i, (user_query, ai_response) in enumerate(st.session_state['chat_history']):
            st.markdown(f"""
                <div class="chat-box">
                    <div class="chat-user">You:</div>
                    <div>{user_query}</div>
                </div>
                <div class="chat-box">
                    <div class="chat-ai">AI:</div>
                    <div>{ai_response}</div>
                </div>
            """, unsafe_allow_html=True)

        # for i, (user_query, ai_response) in enumerate(st.session_state['chat_history']):
        #     st.markdown(f"**You:** {user_query}")
        #     st.markdown(f"**AI:** {ai_response}")

def RAG(docs, query):
    with st.spinner('Processing documents...'):
        for doc in docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as temp_file:
                temp_file.write(doc.read())

        loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text_chunks = text_splitter.split_documents(documents)

        DB_FAISS_PATH = "data/vector_store/faiss_index"
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(text_chunks, embeddings)
        db.save_local(DB_FAISS_PATH)

        llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=db.as_retriever(search_kwargs ={'k':2}),
            return_source_documents = True
        )

        chat_history = []
        result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
        st.success('Document processed successfully!')
        st.session_state['chat_history'].append((query, result['answer']))



if __name__ == '__main__':
    streamlit_ui()


