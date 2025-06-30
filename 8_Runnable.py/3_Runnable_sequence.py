# Runnible have 2 types 
# Taswk specific runnble chat openai prompt templte
# runnable premitive  conect task specific runnables 
# runble lambda runnbale other are  he premitive runiables


# runnbals sequence can abke to conect the two runnbale in a sequesnce 
# in sequesce we can add more as many we awnt

# runnable sequence 

from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model (basic flash version)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

prompt = PromptTemplate(
    template='Write baoutt he joke of any type with topic {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt , model , parser)

result = chain.invoke('summeer')
print(result)


# runnbable patrellel
# upload a pictre or post for same input the put pot is parell and make post for insta and x 



# runbbale pas  through
# input as it si output 




# rnable lambda so we apply any function  




# runnable brach 
# conditon runable (,,,)


# lang chain expression language lcel insead of the runable sequence(--,---,---)
# use pipe |----|---|--|--|