# make a condition review chaecker check the centiment and give the result on the basis of the sentiment as user given 
# if positive then agent give rating bord
# if negatve then we will send a custer help senter

# --- These are conditional chain ----
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal



load_dotenv()

parser = StrOutputParser()

class Validate(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Enter the sentiment: positive, negative, or neutral.")

parse2 = PydanticOutputParser(pydantic_object=Validate)


api_key = os.getenv("GEMINI_API_KEY")
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)


prompt1 = PromptTemplate(
    template="Give me the sentiment of this feedback: {feedback}  \n{format_output}",
    partial_variables={'format_output': parse2.get_format_instructions()},
    input_variables=['feedback']
)



classifier = prompt1 | model1 | parse2

result = classifier.invoke({'feedback': 'The item you guys delivered is astonishingly fantastic and excellent'}).sentiment
print(result)

# ______________________________________________________________________________________________________________________________________________

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()


# ____________________________________________________________________________________________________________ 

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal



load_dotenv()



class Validate(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Enter the sentiment: positive, negative, or neutral.")

parse2 = PydanticOutputParser(pydantic_object=Validate)
parser = StrOutputParser()

api_key = os.getenv("GEMINI_API_KEY")
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)


prompt1 = PromptTemplate(
    template="Give me the sentiment of this feedback: {feedback}  \n{format_output}",
    partial_variables={'format_output': parse2.get_format_instructions()},
    input_variables=['feedback']
)


negative_prompt = PromptTemplate(
    template="Give me the appropriate responce fot negative sentiment {feedback}",
    input_variables=['feedback']

)
positive_prompt = PromptTemplate(
    template="Give me the five star to selk amoung it  {feedback}",
    input_variables=['feedback']
    
)
neutral_prompt =  PromptTemplate(
    template="Give me the appropriate responce fot neutral sentiment {feedback}",
    input_variables=['feedback']
    
)

classifier = prompt1 | model1 | parse2

# Branching


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_prompt | model1 | parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | model1 | parser),
    (lambda x: x.sentiment == 'neutral',  neutral_prompt | model1 | parser),
    RunnableLambda(lambda x: "Could not find any sentiment.")
)

# Full Chain
chain = classifier | branch_chain
print(chain.invoke({'feedback': 'This is a beautiful phone'}))
# # Test