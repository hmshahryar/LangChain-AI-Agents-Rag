from langchain_community.document_loaders import TextLoader

loader = TextLoader('9_Document_loader\data_file.txt' , encoding='utf-8')

docs = loader.load()




from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model (basic flash version)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

prompt = PromptTemplate(
    template='Write a summary fore the following {data}',
    input_variables=['data']
)

parser = StrOutputParser()

chain = prompt | model |  parser
result = chain.invoke({'data':docs[0].page_content})
print(result) 