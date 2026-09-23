from pathlib import Path
from langchain_chroma import Chroma
from src.context import ContextBuilder


class Retriever:
    def __init__(
        self,
        embedding_model,
        collection_name="transformers",
        persist_directory=None,
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or self._default_chroma_path()
        self.context_builder = ContextBuilder()

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model.get_model(),
            persist_directory=str(self.persist_directory),
        )

    def retrieve(self, query, k=3):
        docs = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in docs
        ]

    def build_context(self, results):
        return self.context_builder.build(results)

    @staticmethod
    def _default_chroma_path():
        project_root = Path(__file__).resolve().parents[1]
        return project_root / "chroma_db"


def main():
    from src.embeddings import EmbeddingModel

    embedding_model = EmbeddingModel(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    retriever = Retriever(embedding_model)

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit", "0"}:
            break

        results = retriever.retrieve(query, k=3)

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print(result["text"])


if __name__ == "__main__":
    main()