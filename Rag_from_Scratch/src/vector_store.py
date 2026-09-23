from langchain_chroma import Chroma

class VectorStore:
    def __init__(self,embedding_model):
        self.embedding_model=embedding_model
        self.vectorstore=Chroma(
            collection_name="transformers",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )
        
    def get_config(self):
        return {
            "embedding_model":self.embedding_model
             }
        
        
    def add_documents(self, documents, vectors):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(i) for i in range(len(documents))]

        self.vectorstore._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors
    )
    def count(self):
        return self.vectorstore._collection.count()
    
    def search(self,query_vector,k=3):
        result=self.vectorstore._collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        results=[]
        for i in range(k):
            results.append({
                "text":result["documents"][0][i],
                "metadata":result["metadatas"][0][i],
                "distance":result["distances"][0][i]
            })
        return results