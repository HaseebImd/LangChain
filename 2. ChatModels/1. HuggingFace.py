import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

print("1. Loading environment variables...")
load_dotenv()

# Verify it loaded (prints True if found, False if missing)
print(f"API Key exists in environment: {bool(os.getenv('GOOGLE_API_KEY'))}")

print("2. Initializing model...")
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

print("3. Invoking model (waiting for response)...")
result = model.invoke("What is the capital of Pakistan?")

print("4. Done!")
print(result)
