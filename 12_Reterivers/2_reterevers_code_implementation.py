# from langchain_community.retrievers import WikipediaRetriever
# reteriver = WikipediaRetriever(top_k_results=2 , lang="en")

# query  = "Geological history of pakistan from the prespective of china"
# docs = reteriver.invoke(query) 
# print(docs)# Print retrieved content
# for i, doc in enumerate(docs):
#     print(f"\n--- Result {i+1} ---")
#     print(f"Content:\n{doc.page_content}...")  # truncate for display


# _______________________________________
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Step 3: Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # how many result 

query = "What is Chroma used for?"
results = retriever.invoke(query)

results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# -----------------------------------------
# vector.similarity search 


# --------------------------------------------------------------
# MMr relav ent answer according to the esarch 
# max marginal relevance 
# Sample documents
# docs = [
#     Document(page_content="LangChain makes it easy to work with LLMs."),
#     Document(page_content="LangChain is used to build LLM based applications."),
#     Document(page_content="Chroma is used to store and search document embeddings."),
#     Document(page_content="Embeddings are vector representations of text."),
#     Document(page_content="MMR helps you get diverse results when doing similarity search."),
#     Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
# ]


# __________________________________________________

# broad level question  what i do to ream healthy 

# multi query reteriver 

# query to llm  now k=llm to generate more questin os can aswer more questions 
# and the end all answer are comined 

# ot remove the ambiguity of the question 



# _______________
# contextua compression retereval 
# should be long to get only the desired 

# a para contsin multipele answer  the para contain the exact answer along with un necessay topic 
# so uing context compress reterival  we compress and got only the relavent info 