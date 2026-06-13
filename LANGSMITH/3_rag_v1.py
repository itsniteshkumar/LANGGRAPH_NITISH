import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG chatbot'

load_dotenv()

PDF_PATH = "Varma_Devops.pdf"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

splits = splitter.split_documents(docs)

# 3. Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vs = FAISS.from_documents(splits, embeddings)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. If the answer is not in the context, say 'I don't know'."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])

# 5. Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Ask Questions
print("PDF RAG ready. Ask a question (Ctrl+C to exit)")

while True:
    q = input("\nQ: ")

    if not q.strip():
        continue

    answer = chain.invoke(q)

    print("\nA:", answer)