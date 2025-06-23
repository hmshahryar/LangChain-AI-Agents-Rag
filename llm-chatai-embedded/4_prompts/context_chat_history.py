from langchain_google_genai import GoogleGenerativeAI
from langchain_core import prompts
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
api_of_gemini = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
model = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_of_gemini)
chat_history  = {}
# Greeting
print("🤖 Gemini Chatbot (type 'exit' to quit)\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    chat_history.
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break
    if not user_input:
        print("Please say something!")
        continue
    try:
        result = model.invoke(chat_history)
        chat_history.append(result)
        print("AI:", result)
    except Exception as e:
        print("⚠️ Error:", str(e))
