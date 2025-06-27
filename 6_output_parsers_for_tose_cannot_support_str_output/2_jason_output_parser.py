from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# Create JSON output parser
parser = JsonOutputParser()

# ✅ Correct input_variables list
template1 = PromptTemplate(
    template="Write a detailed report on the topic: {topic} \n\n{formatinstruction}",
    input_variables=["topic"],
    partial_variables={"formatinstruction": parser.get_format_instructions()}
)

# Format prompt
# prompt1 = template1.format(topic="blackhole")

# # Invoke Gemini model
# result1 = model.invoke(prompt1)

# # Show the output
# print(result1.content)
# final_result = parser.parse(result1.content)
# print(final_result)
# print(type(final_result))
#  wii return the jason object and python treat it asd the dict so dict class 


chain =  template1 | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)

# through jason you canto inforce  schema of the outpt parser 
# 
# if want o inform schema of the output or structure  