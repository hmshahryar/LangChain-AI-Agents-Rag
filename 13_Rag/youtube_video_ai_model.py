# -------imports
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)


from youtube_transcript_api import YouTubeTranscriptApi

video_ids = ["Cyv-dgv80kE", "yF9kGESAi3M"]
all_transcripts = []

for video_id in video_ids:
    try:
        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join(chunk["text"] for chunk in transcript_chunks)
        all_transcripts.append(text)
        print(f"✅ Transcript loaded for: {video_id}")
    except Exception as e:
        print(f"❌ Failed to get transcript for {video_id}: {e}")

# full_transcript = "\n\n".join(all_transcripts)
# print("\n📄 Combined Transcript:\n", full_transcript)

