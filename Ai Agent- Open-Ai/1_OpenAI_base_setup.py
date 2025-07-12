from agents import (Agent , GuardrailFunctionOutput , InputGuardrail ,Runner, OpenAIChatCompletionsModel)
import os 
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel , Field


load_dotenv()

api_key_of_gemini = os.getenv("GEMINI_API_KEY")
client = AsyncOpenAI(
    api_key=api_key_of_gemini,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

class Checking_Domain(BaseModel):
    is_home_work : bool = Field( bool , description="is the query is abuot the homw work")
    reason : str = Field(str , description="Give the reason ")


chat_agent = Agent(
    name="general-chat with peoples",
    model=model,
    handoff_description=Field(description="specialised agent for genral purpose query"),
    instructions="you provide answer to user "
)    

answer = Runner.run_sync(chat_agent , "provied the best way to earn money in 2026 with computer science major ")
print(answer.final_output)
