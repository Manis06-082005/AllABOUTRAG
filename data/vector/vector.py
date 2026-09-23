from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()


# 1. Load PDF
loader = PyPDFLoader("Transformers.pdf")

documents = loader.load()

print("Number of documents:", len(documents))


# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store chunks + embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever=vectorstore.as_retriever()
docs=retriever.invoke("what is masked self attention")
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
    print("--------------------")
print("Vector store created successfully!")