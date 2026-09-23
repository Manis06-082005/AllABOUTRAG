
from langchain_community.document_loaders import PyMuPDFLoader
import os 
class DocumentLoader:
    def __init__(self,file_path,file_name):
        self.file_name=file_name
        self.file_path=file_path
    def load(self):
        if os.path.exists(self.file_path):
            
            loader=PyMuPDFLoader(self.file_path)
            documents=loader.load()
            return documents
        raise FileNotFoundError (
            f"file not found:{self.file_path}"
        )   
    def get_info(self):
        return {
            "file_path":self.file_path,
            "file_name":self.file_name
        }
        
# loader=DocumentLoader(
#     './src/Transformers.pdf',
#     'Transformers.pdf'
# )
        
# print(loader.file_name)
# print(loader.file_path)
# documents = loader.load()

# print(len(documents))
# print(documents[0].page_content[:200])
# print(loader.get_info())