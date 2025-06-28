from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini = os.getenv("GEMINI_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=gemini)

# Conversation history
messages = [
    SystemMessage(content="You are a coding specialist who knows how to write efficient, industry-standard code."),
    HumanMessage(content="explai in small para abouts difent all kind of apis")
]

# Invoke the model
result = model.invoke(messages)

# Append the AI's response
messages.append(AIMessage(content=result.content))  # Access the actual text with `.content`

# Print the conversation history
print(messages)
