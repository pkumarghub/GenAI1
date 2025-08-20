import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# This is a simple conversational chain. For a more advanced chatbot,
# you would add features like memory and RAG (Retrieval-Augmented Generation).
# For now, let's just make it a simple chat.

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer user questions based on the conversation history."),
    # ("system", "You are a street dog."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","Question:{question}")
])

## streamlit framework

st.title('Langchain Demo With Gemini API free version')
input_text=st.text_input("Search the topic u want")

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": input_text})

    # Get the response from the model
    response = chain.invoke({
        "question": input_text,
        "chat_history": st.session_state.chat_history
    })
    #
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the response
    st.write(response)

# Display the chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Bot:** {message['content']}")