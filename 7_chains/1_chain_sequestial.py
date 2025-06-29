# chain is epl to make a pipeline and pipe line triger the fort input given 
# then ist step output become the output of the net input 
# chains can help in paperllel propcesisong

# conditional chain and other wasys 

# ! - sequestioal chain
# 2 - parellel chain 
# 3 - conditional chain 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template="Generate the 5 facts about the topic : {topic}",
    input_variables=['topic']

)
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

parser = StrOutputParser()

chian = prompt | model | parser

result = chian.invoke({'topic': 'black hole'})

print(result)

chian.get_graph().print_ascii()