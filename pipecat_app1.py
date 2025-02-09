############DOES NOT WORK################

import os
import shutil
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import TextFrame, EndFrame
from pipecat.processors.frameworks.langchain import LangchainProcessor

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI model and embeddings
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = None  # Placeholder for vectorstore

# Initialize LangchainProcessor
langchain_processor = LangchainProcessor(chain=llm)

# Streamlit UI
st.title("Pipecat PDF RAG Chatbot")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as buffer:
        buffer.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.success("PDF processed and knowledge stored.")

# Pipecat RAG Task
class RAGTask(PipelineTask):
    """Pipecat task to process user queries and generate responses using the RAG model."""
    def __init__(self, pipeline, vectorstore):
        super().__init__(pipeline=pipeline)
        self.vectorstore = vectorstore

    async def run(self, frame, context):
        if isinstance(frame, TextFrame):
            question = frame.text
            retriever = self.vectorstore.as_retriever()
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Prepare the prompt for the Langchain model
            prompt = {
                "context": context_text,
                "question": question
            }

            # Use the LangchainProcessor to handle the prompt
            response = await langchain_processor.process(prompt)
            return TextFrame(response)
        return EndFrame()

# Query Interface
if vectorstore:
    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer") and question:
        pipeline = Pipeline()
        rag_task = RAGTask(pipeline=pipeline, vectorstore=vectorstore)
        pipeline.add_task(rag_task)
        
        input_frame = TextFrame(text=question)
        response_frame = pipeline.run(input_frame)
        
        if isinstance(response_frame, TextFrame):
            st.write("### Answer:")
            st.write(response_frame.text)
else:
    st.warning("Please upload a PDF first.")