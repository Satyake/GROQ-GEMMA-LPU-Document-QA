import os
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Q and A")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only,
    provide accurate response based on question only
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader('./data')
        st.session_state.docs = st.session_state.loader.load()  # load all documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, embedding=st.session_state.embeddings)
        st.session_state.vectors = st.session_state.vectorstore  # initialize vectors in session state

prompt1 = st.text_input("Enter your Question from the documents?")
if st.button("Creating Vectorstore"):
    vector_embedding()
    st.write("Vectorstore created")

import time  

if prompt1:
    document_chain = create_stuff_documents_chain(
        llm, prompt 
    )

    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever, document_chain
    )
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------")
