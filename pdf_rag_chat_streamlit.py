import streamlit as st
from dotenv import load_dotenv
import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("PDF-based RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    document = loader.load()

    # Split PDF into chunks and get embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Use FAISS for vector storage
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # RAG Prompt Setup
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # RAG Chain Setup
    def rag_chat(question):
        # Retrieve relevant context
        retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)

        # Format chat history
        formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in st.session_state.chat_history])

        input_data = {
            "context": context,
            "question": question,
            "history": formatted_history
        }

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )

        response = rag_chain.invoke(input_data)

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        return response

    # Chat interface
    st.subheader("Chat with your PDF")
    user_input = st.text_input("Ask a question about the document:")

    if st.button("Ask") and user_input:
        response = rag_chat(user_input)
        st.write(f"**AI:** {response}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for entry in st.session_state.chat_history:
            role = "User" if entry["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {entry['content']}")
