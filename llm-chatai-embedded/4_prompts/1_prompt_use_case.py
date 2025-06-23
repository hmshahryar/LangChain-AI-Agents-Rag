from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
api_of_gemini = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" , api_key=api_of_gemini)
user_input = st.text_input("Enter your Query to get responce : ")

if st.button("Let's do it "):
    result = model.invoke(user_input)
    st.write(result.content)