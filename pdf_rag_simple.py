from dotenv import load_dotenv
import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
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

# Get PDF filename and question dynamically
pdf_path = input("Enter the PDF filename (including extension): ").strip()
question = input("Enter your question: ").strip()

# Load and process PDF
loader = PyPDFLoader(pdf_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # Using OpenAI's embeddings

# Use FAISS for vector storage
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# RAG Prompt Setup
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Get responses
print(f"\nQuestion: {question}")
print("\nAnswer without RAG:")
try:
    print(llm.invoke([question]).content)
except Exception as e:
    print(f"Error invoking LLM: {e}")

print("\nAnswer with RAG:")
try:
    print(rag_chain.invoke(question))
except Exception as e:
    print(f"Error invoking RAG chain: {e}")
