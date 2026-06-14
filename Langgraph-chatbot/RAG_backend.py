from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
import torch
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==========================================================
# Environment
# ==========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ==========================================================
# LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ==========================================================
# Embeddings
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Starting RAG Backend")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")

print(f"Embedding Device : {DEVICE}")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device": DEVICE,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 16,
    },
)

# Validate model loads successfully

try:
    test_vector = embeddings.embed_query("hello world")
    print(f"Embedding dimension = {len(test_vector)}")
except Exception as e:
    print("Failed to initialize embeddings")
    raise e

# ==========================================================
# PDF Retriever Store
# ==========================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Build FAISS index for uploaded PDF.
    """

    if not file_bytes:
        raise ValueError("No bytes received.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
            },
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ==========================================================
# Tools
# ==========================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Calculator tool.

    Supported:
    add
    sub
    mul
    div
    """

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation: {operation}"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price using Alpha Vantage.
    """

    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey=C9PE94QUEW9VWGFM"
        )

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:
        return {
            "error": str(e)
        }


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Retrieve relevant content from uploaded PDF.
    """

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    docs = retriever.invoke(query)

    return {
        "query": query,
        "chunks": [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ],
        "source_file": _THREAD_METADATA.get(
            str(thread_id),
            {},
        ).get("filename"),
    }


# ==========================================================
# Tool Binding
# ==========================================================

tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)

# ==========================================================
# State
# ==========================================================


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# Chat Node
# ==========================================================


def chat_node(
    state: ChatState,
    config=None,
):
    thread_id = None

    if config and isinstance(config, dict):
        thread_id = (
            config.get("configurable", {})
            .get("thread_id")
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful RAG assistant.\n\n"
            "Rules:\n"
            "1. If user asks about uploaded PDF ALWAYS call rag_tool.\n"
            "2. Include current thread_id in rag_tool.\n"
            "3. Answer ONLY from retrieved PDF context.\n"
            "4. If answer isn't present, say so.\n"
            "5. Mention page numbers when available.\n"
            "6. Use search, calculator and stock tools when helpful.\n\n"
            f"Current thread_id: {thread_id}"
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [response]
    }


# ==========================================================
# Graph
# ==========================================================

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

# ==========================================================
# Checkpointer
# ==========================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(
    checkpointer=checkpointer
)

# ==========================================================
# Helper Methods
# ==========================================================


def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)


def thread_has_document(
    thread_id: str,
) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(
    thread_id: str,
) -> dict:
    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )