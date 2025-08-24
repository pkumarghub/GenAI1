import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# load the groq API key
groq_api_key = os.environ["GROQ_API_KEY"]
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

# Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if "vectors" not in st.session_state:
    st.session_state.embeddings = embeddings
    st.session_state.loader = WebBaseLoader(
        "https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings)
    
st.title("Groq Demo")

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
llm = ChatGroq(groq_api_key=groq_api_key, model_name = "deepseek-r1-distill-llama-70b")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input you prompt here")

if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("document similarty search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------------") 
