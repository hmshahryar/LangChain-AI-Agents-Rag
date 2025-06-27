from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
import os
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

class know(BaseModel):

    name : str = Field(description='Write the name fof the person ')
    age : int = Field(description='Age of the person' , gt=18)
    city : str = Field(description='write cithy of the user '
                       )

parser = PydanticOutputParser(pydantic_object=know)


template1 = PromptTemplate(
    template="generate the age name city of the fiction person : {topic} \n\n{formatinstruction}",
    input_variables=["topic"],
    partial_variables={"formatinstruction": parser.get_format_instructions()}
)



chain =  template1 | model | parser
result = chain.invoke({'topic':'pakistani'})
print(result)

