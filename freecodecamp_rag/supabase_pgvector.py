import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# Supabase
# =========================

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# =========================
# Load PDF
# =========================

loader = PyMuPDFLoader("Rag.pdf")
documents = loader.load()

print("Pages:", len(documents))


# =========================
# Text Splitting
# =========================

def text_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks


# =========================
# Embeddings
# =========================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chunks = text_splitter()
texts = [chunk.page_content for chunk in chunks]

embed = embedding.embed_documents(texts)


# =========================
# Store Documents
# =========================

def store_documents(chunks, embeddings):
    data = []

    for chunk, vector in zip(chunks, embeddings):
        records = {
            "embedding": vector,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }

        data.append(records)

    response = supabase.table("documents").insert(data).execute()


store_documents(chunks, embed)


# =========================
# Retrieval
# =========================

def search_documents(query, k=5):
    query_vector = embedding.embed_query(query)

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "match_count": k
        }
    ).execute()

    return result.data


# Test Retrieval

results = search_documents("What is RAG")

for result in results:
    print(result["content"])
    print("Similarity:", result["similarity"])
    print()


# =========================
# LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# =========================
# Generation
# =========================

def generate_answers(query, results):
    content = [result["content"] for result in results]
    context = "\n\n".join(content)

    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{content}

Question:
{query}

If you don't know the answer, say "I don't know."
""")

    res = prompt | llm

    answer = res.invoke({
        "query": query,
        "content": context
    })

    return answer.content


# =========================
# Complete RAG Pipeline
# =========================

result = search_documents("What is RAG", 3)
answer = generate_answers("What is RAG", result)

print(answer)