from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

prompt = PromptTemplate(
    template="explain me the topic : {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summurize it in 5 points the topic {topic}"
    , input_variables=['topic']
)
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'how to make 1 million dolar'})

print(result)
chain.get_graph().print_ascii()