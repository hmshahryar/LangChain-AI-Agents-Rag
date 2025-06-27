from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
import os
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)


schema = [
    ResponseSchema(name='fact1' , description='fact1 about the topic'),
    ResponseSchema(name='fact2' , description='fact1 about the topic'),
    ResponseSchema(name='fact3' , description='fact1 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)


template1 = PromptTemplate(
    template="Write fsctd  report on the topic: {topic} \n\n{formatinstruction}",
    input_variables=["topic"],
    partial_variables={"formatinstruction": parser.get_format_instructions()}
)



chain =  template1 | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)

# through jason you canto inforce  schema of the outpt parser  
# if want o inform schema of the output or structure  use struct output parser