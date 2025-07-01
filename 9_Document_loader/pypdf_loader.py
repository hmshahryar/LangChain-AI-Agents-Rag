from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 


doc = PyPDFLoader(file_path='Pictures_Notes\\the-art-of-seduction-robert-greene.pdf')


docs = doc.load()
print(len(docs))
print(docs[1].metadata)

# api_key = os.getenv("GEMINI_API_KEY")

# # Initialize Gemini model (basic flash version)
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# prompt = PromptTemplate(
#     template='Write a summary fore the following {data}',
#     input_variables=['data']
# )