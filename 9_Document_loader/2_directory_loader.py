from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

dir_loader = DirectoryLoader(
    path='Pictures_Notes' ,  # folder path ,
    glob='**/*.pdf' ,  # which file patter you wan to upload 
    loader_cls = PyPDFLoader
)
# lazy loading document memory load giv generator of doc and upload oe in memory 
# lazy putr signgle single on demand give generator of docs 

# loa ios eafer load 

# lazy loader good for large context 

documnet = dir_loader.load()
print(len(documnet))

api_key = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

prompt = PromptTemplate(
    template='Write a summary fore the following {data}',
    input_variables=['data']
)