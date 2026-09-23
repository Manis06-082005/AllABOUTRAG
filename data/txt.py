from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings


loader=TextLoader(
    'example.txt',
    encoding='utf-8'
)
document=loader.load()
print(document[0].metadata)

loader=PyPDFLoader(
    'Transformers.pdf'
)
doc2=loader.load()
print(doc2[7
           ].page_content)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=5
    
    
)

# chunks=splitter.split_documents(document
#                            )
# print("Number of chunks:", len(chunks))
# print(chunks[0])
# splitter_2=CharacterTextSplitter(
#     separator='/n/n',
#     chunk_size=200,
#     chunk_overlap=10
# )
# chunks=splitter_2.split_documents(document)
# print("Number of chunks:", len(chunks))
# print(chunks[0])
splitter_3=TokenTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks=splitter_3.split_documents(document)
print(len(chunks))
print(chunks[0])



# Semantic Meaning-Based Splitting

# Definition: Semantic splitting divides a document into chunks based on changes in meaning/topic, rather than simply cutting every 500 characters or 200 tokens.

# In LangChain, you can use SemanticChunker for this.
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings

# 1. Load document


# 2. Create embeddings
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

# 3. Create semantic splitter
splitter = SemanticChunker(embeddings)

# 4. Split based on meaning
chunks = splitter.split_documents(document)

# 5. Check result
print("Documents:", len(document))
print("Chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

