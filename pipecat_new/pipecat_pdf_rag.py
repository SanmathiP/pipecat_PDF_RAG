import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize Markdown Text Filter
text_filter = MarkdownTextFilter(
    params=MarkdownTextFilter.InputParams(
        enable_text_filter=True,
        filter_code=True,
        filter_tables=True
    )
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("Pipecat X PDF-based RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    document = loader.load()

    # Split PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Use FAISS for vector storage
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join([text_filter.filter(doc.page_content) for doc in docs[:2]])  # Ensure proper newlines

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # RAG Chain Setup
    def rag_chat(question):
        # Retrieve relevant context
        retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)

        # Format chat history
        formatted_history = "\n\n".join([f"{turn['role']}: {turn['content']}" for turn in st.session_state.chat_history])

        input_data = f"Context:\n{context}\n\nQuestion:\n{question}\n\nChat History:\n{formatted_history}"

        # Generate response using LLM
        response = llm.invoke(input_data)

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        return response

    # Chat interface
    st.subheader("Chat with your PDF")
    user_input = st.text_input("Ask a question about the document:")

    if st.button("Ask") and user_input:
        response = rag_chat(user_input)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_metadata = response.response_metadata if hasattr(response, 'response_metadata') else {}
        response_usage = response.usage_metadata if hasattr(response, 'usage_metadata') else {}

        st.write(f"**AI:** {response_text}\n\n")  # Ensure newline spacing
        st.write(f"additional_kwargs:\n{response.additional_kwargs if hasattr(response, 'additional_kwargs') else {}}\n\n")
        st.write(f"response_metadata:\n{response_metadata}\n\n")
        st.write(f"usage_metadata:\n{response_usage}\n\n")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for entry in st.session_state.chat_history:
            role = "User" if entry["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {entry['content']}\n\n")
