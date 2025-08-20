import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables
load_dotenv()

# Validate required environment variables
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer user questions based on the conversation history."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Question: {question}")
])

# Streamlit UI setup
st.title("LangChain Demo with Gemini API")
input_text = st.text_input("Enter your query:")

# Create the chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Process user input
if input_text:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": input_text})

    try:
        # Get the response from the model
        response = chain.invoke({
            "question": input_text,
            "chat_history": st.session_state.chat_history
        })

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display the response
        st.write(f"**Bot:** {response}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Bot:** {message['content']}")