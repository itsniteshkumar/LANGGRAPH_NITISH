import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# Environment Variables
# ==================================================

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_PATH = "Varma_Devops.pdf"
FAISS_INDEX_PATH = "faiss_index"


# ==================================================
# Load PDF
# ==================================================

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


# ==================================================
# Split Documents
# ==================================================

@traceable(name="split_documents")
def split_documents(
    docs,
    chunk_size=1000,
    chunk_overlap=150,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splits = splitter.split_documents(docs)

    print(f"\nTotal chunks created: {len(splits)}")

    return splits


# ==================================================
# Gemini Embeddings
# ==================================================

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


# ==================================================
# Build Vector Store
# ==================================================

@traceable(name="build_vectorstore")
def build_vectorstore(splits):

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    return vectorstore


# ==================================================
# Setup Pipeline
# ==================================================

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vectorstore = build_vectorstore(splits)

    vectorstore.save_local(FAISS_INDEX_PATH)

    print("\nFAISS index saved successfully.")

    return vectorstore


# ==================================================
# Load/Create Vector Store
# ==================================================

embeddings = get_embeddings()

if os.path.exists(FAISS_INDEX_PATH):

    print("\nLoading existing FAISS index...")

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

else:

    print("\nCreating new FAISS index...")

    vectorstore = setup_pipeline(PDF_PATH)


# ==================================================
# Retriever
# ==================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# ==================================================
# Gemini LLM
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ==================================================
# Prompt
# ==================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.

Answer ONLY from the supplied context.

If the answer is not present in the context,
respond with:

"I don't know based on the provided document."
""",
        ),
        (
            "human",
            """Question:
{question}

Context:
{context}
""",
        ),
    ]
)


# ==================================================
# Format Retrieved Docs
# ==================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==================================================
# RAG Chain
# ==================================================

parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# Chat Loop
# ==================================================

print("\nPDF RAG Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("Q: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    config = {
        "run_name": "pdf_rag_query"
    }

    try:

        answer = chain.invoke(
            question,
            config=config,
        )

        print("\nA:", answer)
        print("\n" + "=" * 80 + "\n")

    except Exception as e:

        print("\nERROR:")
        print(e)