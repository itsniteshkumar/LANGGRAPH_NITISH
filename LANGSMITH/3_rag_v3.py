import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# ==================================================
# Environment
# ==================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_PATH = "Varma_Devops.pdf"

os.environ["LANGCHAIN_PROJECT"] = "PDF RAG Gemini"


# ==================================================
# Load PDF
# ==================================================

@traceable(name="load_pdf", tags=['pdf', 'loader'], metadata={'loader': 'PyPDFLoader'})
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
    chunk_overlap=150
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = splitter.split_documents(docs)

    print(f"Total Chunks: {len(splits)}")

    return splits


# ==================================================
# Embeddings
# ==================================================

def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


# ==================================================
# Build Vector Store
# ==================================================

@traceable(name="build_vectorstore", tags=['embeddings', 'vectorestore'], metadata={'embedding-model': 'gemini-embedding-001'})
def build_vectorstore(splits):

    embeddings = get_embeddings()

    return FAISS.from_documents(
        splits,
        embeddings
    )


# ==================================================
# Parent Setup Function
# ==================================================

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    pdf_path: str,
    chunk_size=1000,
    chunk_overlap=150
):

    docs = load_pdf(pdf_path)

    splits = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    vs = build_vectorstore(splits)

    return vs


# ==================================================
# Gemini LLM
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ==================================================
# Prompt
# ==================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer ONLY from the provided context.

If the answer is not present in the context,
say:

I don't know based on the provided document.
"""
    ),
    (
        "human",
        """Question:
{question}

Context:
{context}
"""
    )
])


# ==================================================
# Helper
# ==================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ==================================================
# Root Run
# ==================================================

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str
):

    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size=1000,
        chunk_overlap=150
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = (
        parallel
        | prompt
        | llm
        | StrOutputParser()
    )

    lc_config = {
        "run_name": "pdf_rag_query"
    }

    return chain.invoke(
        question,
        config=lc_config
    )


# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    print("PDF RAG Ready (Gemini)")
    print("Press Ctrl+C to Exit\n")

    q = input("Q: ").strip()

    ans = setup_pipeline_and_query(
        PDF_PATH,
        q
    )

    print("\nA:", ans)