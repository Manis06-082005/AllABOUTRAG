import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from src.embeddings import EmbeddingModel
from src.indexer import Indexer
from src.retrieval import Retriever
from src.generator import Generator

load_dotenv()

embedding_model = EmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2"
)

indexer = Indexer(
    "./src/Transformers.pdf",
    "Transformers.pdf",
    500,
    50,
    "sentence-transformers/all-MiniLM-L6-v2",
    embedding_model
)

indexer.index()

retriever = Retriever(embedding_model)

generator = Generator("gemini-3.6-flash")

while True:
    query = input("\nQuestion: ").strip()

    if query.lower() in {"exit", "quit", "0"}:
        break

    results = retriever.retrieve(query, k=3)
    context = retriever.build_context(results)

    answer = generator.generate(query, context)

    print("\nAnswer:", answer)
    