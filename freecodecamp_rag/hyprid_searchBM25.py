import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
PDF_PATH = "Rag.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

BM25_K = 10
VECTOR_K = 10
FUSION_K = 20
RERANK_K = 10
TOKEN_BUDGET = 3000

CHROMA_DIR = "./chroma_db"


def create_chunks():
    documents=PyMuPDFLoader(PDF_PATH).load()
    splitter=RecursiveCharacterTextSplitter(
        CHUNK_OVERLAP,
        CHUNK_SIZE
    )
    return splitter.split_documents(
        documents
    )

def create_vector_store(chunks,embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hybrid-rag",
        persist_directory=CHROMA_DIR
    )
    
def create_bm25_retriever(chunks):
    return BM25Retriever.from_documents(
        chunks,
        k=BM25_K
    )

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def reciprocal_rank_fusion(result_lists, k=60, top_n=20):
    scores = {}
    documents = {}

    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            doc_id = (
                doc.metadata.get("source", "")
                + str(doc.metadata.get("page", ""))
                + doc.page_content
            )

            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:top_n]

    return [documents[i] for i in ranked_ids]

def rerank_documents(query, docs, reranker):
    pairs = [[query, doc.page_content] for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:RERANK_K]]

encoding = tiktoken.get_encoding("cl100k_base")


def apply_token_budget(docs, max_tokens=TOKEN_BUDGET):
    selected = []
    used_tokens = 0

    for doc in docs:
        tokens = len(encoding.encode(doc.page_content))

        if used_tokens + tokens <= max_tokens:
            selected.append(doc)
            used_tokens += tokens

    return selected, used_tokens
def format_context(docs):
    return "\n\n".join(
        f"[Source {i} | Page {doc.metadata.get('page', 'unknown')}]\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )
def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )
class HybridRAG:

    def __init__(self):
        print("Initializing Hybrid RAG...")

        self.chunks = create_chunks()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = create_vector_store(
            self.chunks,
            self.embeddings
        )

        self.bm25_retriever = create_bm25_retriever(
            self.chunks
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.llm = create_llm()

        print("Hybrid RAG ready.")

    def query(self, query):
        bm25_docs = self.bm25_retriever.invoke(query)

        vector_docs = self.vector_store.similarity_search(
            query,
            k=VECTOR_K
        )

        fused_docs = reciprocal_rank_fusion(
            [bm25_docs, vector_docs],
            top_n=FUSION_K
        )

        reranked_docs = rerank_documents(
            query,
            fused_docs,
            self.reranker
        )

        final_docs, used_tokens = apply_token_budget(
            reranked_docs
        )

        context = format_context(final_docs)

        prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response, final_docs, used_tokens
if __name__ == "__main__":

    rag = HybridRAG()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        response, docs, used_tokens = rag.query(query)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(response.content)

        print("\n" + "=" * 60)
        print("TOKEN BUDGET")
        print("=" * 60)
        print(f"Used: {used_tokens}/{TOKEN_BUDGET}")

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        for i, doc in enumerate(docs, 1):
            print(
                f"{i}. {doc.metadata.get('source', 'unknown')} "
                f"| Page {doc.metadata.get('page', 'unknown')}"
            ) 
