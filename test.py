from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


llm = ChatHuggingFace(repo_id="TheBloke/vicuna-7B-1.1-HF", task="text-generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is Capital of Pakistan?")
print(result)
