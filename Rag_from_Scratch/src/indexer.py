from src.loader import DocumentLoader
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.chunker import TextChunker


class Indexer:
    def __init__(
        self,
        file_path,
        file_name,
        chunk_size,
        chunk_overlap,
        model_name,
        embedding_model=None,
    ):
        self.loader = DocumentLoader(file_path, file_name)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.vector_store = VectorStore(model_name)

    def index(self):
        if self.vector_store.count() == 0:
            documents = self.loader.load()
            chunks = self.chunker.split(documents)

            texts = [chunk.page_content for chunk in chunks]

            if self.embedding_model is None:
                self.embedding_model = EmbeddingModel(self.model_name)

            vectors = self.embedding_model.embed_documents(texts)
            self.vector_store.add_documents(chunks, vectors)

            return len(chunks)

        return 0