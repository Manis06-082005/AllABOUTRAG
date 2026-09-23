from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
from src.chunker import TextChunker
from src.loader import DocumentLoader
class EmbeddingModel:
    def __init__(self,model_name):
        self.model_name=model_name
        self.model=HuggingFaceEmbeddings(
            model_name=model_name
        )
    def get_model(self):
        return self.model

    def embed_text(self,text):
        return self.model.embed_query(text)
    
    def embed_documents(self,documents):
        return self.model.embed_documents(documents)
    
    
# model = EmbeddingModel(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )
# embedding_model = model.get_model()
# vector=model.embed_text('Hi AI')
# print(type(vector))
# print(len(vector))
# print(vector[:5])

# print(model.model_name)
# print(type(embedding_model))
# doc=model.embed_documents([
#     "What is attention?",
#     "Transformers use self attention.",
#     "Neural networks learn representations."
# ])
# print(len(doc))
# print(len(doc[0]))