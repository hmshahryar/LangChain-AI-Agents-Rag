from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get token
huggingface_api_key = os.getenv("HUGGINGFAVEHUB_ACCESS_TOKEN")

# Debug check
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Please check your .env file.")

# Load model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=huggingface_api_key,
)

# Use model
prompt = "How can I make 1 million dollars in a year with zero base cash?"
response = llm.invoke(prompt)
print(response)
