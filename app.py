from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatOllama(model="llama3.2:latest")

response = llm.invoke("Explain Kubernetes")

print(response.content)