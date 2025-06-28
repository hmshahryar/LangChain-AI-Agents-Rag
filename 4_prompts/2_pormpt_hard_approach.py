from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os

# Load environment variables
load_dotenv()
api_of_gemini = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_of_gemini)

# Streamlit inputs
st.title("📝 Creative Poem Generator")

style = st.selectbox("Choose a poem style:", ["haiku", "sonnet", "limerick", "free verse"])
language = st.selectbox("Choose a language:", ["English", "Urdu", "Spanish", "French"])
tone = st.selectbox("Choose a tone:", ["romantic", "sad", "funny", "inspirational", "mystical"])

topic = st.text_input("Enter the topic of the poem:")
reference = st.text_input("Enter references (e.g. moon, sea, stars):")

# Define prompt template
template_base = """
You are a creative poet. Write a {style} poem about {topic} in {language}.
The tone should be {tone}, and it should include references to {reference}.
"""

templatee = PromptTemplate(
    template=template_base,
    input_variables=["style", "topic", "language", "reference", "tone"]
)

# Button click logic
if st.button("Let's do it"):
    if topic.strip() and reference.strip():
        # Format prompt
        filled_prompt = templatee.format(
            style=style,
            topic=topic,
            language=language,
            tone=tone,
            reference=reference
        )

        # Call Gemini model
        result = model.invoke(filled_prompt)

        # Display result
        st.subheader("🎨 Generated Poem:")
        st.write(result.content if hasattr(result, "content") else result)
    else:
        st.warning("Please enter both a topic and a reference!")
