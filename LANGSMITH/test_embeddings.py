from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector = embeddings.embed_query("Hello world")

print(f"Embedding length: {len(vector)}")
print(vector[:5])