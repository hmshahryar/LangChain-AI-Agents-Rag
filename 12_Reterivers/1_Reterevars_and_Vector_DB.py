# In LangChain, retrievers are components that fetch relevant documents from a knowledge base (like a vector database or a text index) in response to a query. 


# reterevers are runable 
# so can use in chains  
# and plugin reterevers can be places in cchin 


# difent reterivcal use diffent mechnaismm
# dfent reterevas data store type reterivers 

# reerivers on whcich sorce we are using reterivers like data source retervers 
# how it work on the basis of way it work the reterval work 

# runnables are those who have invoke function capability 

# ------------------------------------------------------
# ✅ LangChain + ChromaDB: PDF Loader & Persistent Vector Store
# ------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from chromadb import PersistentClient
import os

# 1. Load PDF using LangChain's PyPDFLoader
loader = PyPDFLoader("Pictures_Notes/Explanation-of-Topics.pdf")
pages = loader.load()

# 2. Extract text content and create unique IDs
documents = [page.page_content for page in pages]
ids = [f"id_{i}" for i in range(len(documents))]

# 3. Create a directory to store the persistent Chroma database
persist_dir = "13_Persist_Directory_chroma_store"
os.makedirs(persist_dir, exist_ok=True)

# 4. Initialize Persistent Chroma Client (✅ CORRECT WAY)
from chromadb.config import Settings

chroma_client = PersistentClient(
    path=persist_dir,
    settings=Settings(
        anonymized_telemetry=False  # optional
    )
)

# 5. Create or get a collection
collection = chroma_client.get_or_create_collection(name="shari")

# 6. Add documents (use `upsert` to avoid duplicates)
collection.upsert(
    ids=ids,
    documents=documents
)

# 7. Done
print("✅ Documents successfully stored in:", persist_dir)

# Optional: Query for testing
results = collection.query(
    query_texts=["Tell me about pineapple"],  # Replace with any question
    n_results=2
)

print("🔍 Query Results:")
print(results)


# _________________________
# Reconnect to Chroma
# chroma_client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./my_chroma_data"
# ))

# # Reuse your collection
# collection = chroma_client.get_collection(name="my_pdfs")

# # Search or query
# results = collection.query(query_texts=["Tell me about pineapple"], n_results=2)
# print(results)
