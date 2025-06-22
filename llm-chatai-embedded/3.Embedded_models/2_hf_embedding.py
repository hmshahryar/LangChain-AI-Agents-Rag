from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
# Get token
huggingface_api_key = os.getenv("HUGGINGFAVEHUB_ACCESS_TOKEN")
# Load the embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text to embed
text = "delhi capital"

# Get the embedding vector
vector = embeddings.embed_query(text)

# Print the vector
print(vector)
