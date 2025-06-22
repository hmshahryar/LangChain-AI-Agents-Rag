from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if you have one)
load_dotenv()

# Retrieve your GEMINI_API_KEY from environment variables
# Make sure GEMINI_API_KEY is set in your environment or a .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")

# Initialize the GoogleGenerativeAIEmbeddings with your API key
# You specify the model name here.
# Note: "gemini-embedding-exp-03-07" might be an experimental model name.
# For general use, "gemini-embedding-001" is a stable option.
# I'll use "gemini-embedding-001" as a common example.
# If you specifically need "gemini-embedding-exp-03-07", ensure it's available and compatible.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", # Or "models/gemini-embedding-exp-03-07" if that's what you specifically need
    google_api_key=GEMINI_API_KEY
)

# Use the embed_query method to get the embedding for a single string
query_text = "What is the meaning of life?"
result_embedding = embedding_model.embed_query(query_text)

# The result_embedding will be a list of floats (the embedding vector)
print(result_embedding)

# If you were embedding multiple documents:
# documents = ["Document 1 text here.", "Document 2 text here."]
# document_embeddings = embedding_model.embed_documents(documents)
# print(document_embeddings)