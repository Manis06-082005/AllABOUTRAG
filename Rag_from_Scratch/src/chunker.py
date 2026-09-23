from langchain_text_splitters import RecursiveCharacterTextSplitter
# from src.loader import DocumentLoader
# from src.embeddings import EmbeddingModel
class TextChunker:
    def __init__(self,chunk_size,chunk_overlap):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=self.chunk_size,
    chunk_overlap=self.chunk_overlap
)
            
    def split(self,documents):
        return self.splitter.split_documents(documents)
    def get_config(self):
        return {
    "chunk_size": self.chunk_size,
    "chunk_overlap": self.chunk_overlap
}
    def get_chunk_info(self, chunks):
        return{
            "total_length":len(chunks),
            "content":chunks[0].page_content,
            "content_length":len(chunks[0].page_content),
            "last_chunk":len(chunks[-1].page_content)
        }
# chunker=TextChunker(500,50)
# loader=DocumentLoader(
#     './src/Transformers.pdf',
#     'Transformers.pdf'
# )
# documents=loader.load()
# chunks=chunker.split(documents)
# print(chunker.get_chunk_info(chunks))
# print(len(chunks))
# texts = [chunk.page_content for chunk in chunks]
# metadatas = [chunk.metadata for chunk in chunks]
# model = EmbeddingModel(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )
# vector=model.embed_documents(texts)

