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
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Get PDF filename dynamically
pdf_path = input("Enter the PDF filename (including extension): ").strip()

# Load and process PDF
loader = PyPDFLoader(pdf_path)
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

# Conversation history
chat_history = []

# RAG Chain Setup
def rag_chat(question):
    global chat_history

    # Retrieve relevant context
    retrieved_docs = retriever.invoke(question)

    # Format context into a single string
    context = format_docs(retrieved_docs)

    # Format chat history
    formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])

    # Prepare input dictionary
    input_data = {
        "context": context,
        "question": question,
        "history": formatted_history
    }

    # Corrected Runnable Sequence
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # Get response
    response = rag_chain.invoke(input_data)

    # Update chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response})

    return response

# Chat loop
print("\nStart chatting! Type 'exit' to stop.\n")
while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Ending conversation. Goodbye!")
        break
    response = rag_chat(question)
    print(f"AI: {response}\n")
