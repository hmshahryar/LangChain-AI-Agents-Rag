#4 types of outputparsers 
#1 - string
#2 - json
#3- structure
#4 - pydantic
# str output parser related with chains whains help to mange in a pipe line  
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model (basic flash version)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# Define prompt templates
template1 = PromptTemplate.from_template("Write a detailed report on the topic: {topic}")
template2 = PromptTemplate.from_template("Write a 5-line summary of the following report:\n\n{report}")

# Step 1: Detailed Report
prompt1 = template1.format(topic="blackhole")
result1 = model.invoke(prompt1)

# Step 2: Summary based on generated report
prompt2 = template2.format(report=result1.content)
result2 = model.invoke(prompt2)

# ------------------
prser = StrOutputParser()
chain = template1 | model | prser | template2 |model | prser 
result = chain.invoke({'topic': 'black hole'})
# Print the final summary
print(result2.content)

