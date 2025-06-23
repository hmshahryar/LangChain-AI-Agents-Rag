from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
api_of_gemini = os.getenv("GEMINI_API_KEY")

# Load Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_of_gemini)

# Streamlit UI
st.title("📝 Creative Poem Generator")

style = st.selectbox("Choose a poem style:", ["haiku", "sonnet", "limerick", "free verse"])
language = st.selectbox("Choose a language:", ["English", "Urdu", "Spanish", "French"])
tone = st.selectbox("Choose a tone:", ["romantic", "sad", "funny", "inspirational", "mystical"])
topic = st.text_input("Enter the topic of the poem:")
reference = st.text_input("Enter references (e.g. moon, sea, stars):")

# Prompt Template
prompt = PromptTemplate.from_template("""
You are a creative poet. Write a {style} poem about {topic} in {language}.
The tone should be {tone}, and it should include references to {reference}.
""")

# Create chain using pipe operator
chain = prompt | model

# Run on button click
if st.button("Let's do it"):
    if topic.strip() and reference.strip():
        
        result = chain.invoke({
            "style": style,
            "topic": topic,
            "language": language,
            "tone": tone,
            "reference": reference
        }
)

        # Display result
        st.subheader("🎨 Generated Poem:")
        st.write(result.content if hasattr(result, "content") else result)
    else:
        st.warning("Please fill in all fields!")
