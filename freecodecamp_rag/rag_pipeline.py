from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel,fields
from typing import List
import tempfile
from  langchain_community.document_loaders import PyMuPDFLoader

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
embedding_model=HuggingFaceEmbeddings()
knowledge_base="ai-report.pdf"

llm=init_chat_model(
    model="gemini-3.6-flash",
    temprature=0.2    
)

def create_kb():
    """Create a vector store from knowledge base."""

    # split the knowledge base into chunks
    loader=PyMuPDFLoader(knowledge_base)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc=loader.load()

    chunks = splitter.split_documents(doc)

    # create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def basic_rag():
    vector_store=create_kb()
    retriever=vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k":3}
    )
    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
