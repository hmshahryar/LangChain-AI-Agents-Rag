from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os
google_api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash" , api_key=google_api_key , temperature=0.8 , max_tokens=50 , )

result = model.invoke("suggest me the best way so i can make alot of monaey i am earnig the aiagents and i knows the front end using react and backend through nextjs and ts")
print(result.content)