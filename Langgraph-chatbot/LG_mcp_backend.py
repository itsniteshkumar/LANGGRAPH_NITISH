from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

import aiosqlite
import requests
import asyncio
import threading
import os
import sys

load_dotenv()

# ============================================================
# Async Event Loop
# ============================================================

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True
)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP
    )


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)

# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

# ============================================================
# Local Tools
# ============================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get latest stock price using AlphaVantage.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey=C9PE94QUEW9VWGFM"
    )

    response = requests.get(url, timeout=20)
    return response.json()

# ============================================================
# MCP
# ============================================================

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                r"C:\Users\nitesh\Desktop\CODE\REPOSITORY_CODE\LANGGRAPH\Langgraph-chatbot\mcp_math.py"
            ],
        },

        # Disabled until schema is fixed
        # "expense": {
        #     "transport": "streamable_http",
        #     "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        # }
    }
)

# ============================================================
# MCP Loader
# ============================================================


def load_mcp_tools() -> list[BaseTool]:
    try:

        tools = run_async(client.get_tools())

        valid_tools = []

        for tool in tools:

            try:
                print(f"Found MCP Tool: {tool.name}")

                schema = getattr(tool, "args", {})

                if not isinstance(schema, dict):
                    valid_tools.append(tool)
                    continue

                properties = schema.get(
                    "properties",
                    {}
                )

                invalid = False

                for _, value in properties.items():
                    if value is None:
                        invalid = True
                        break

                if invalid:
                    print(
                        f"Skipping invalid MCP tool: {tool.name}"
                    )
                    continue

                valid_tools.append(tool)

            except Exception as e:
                print(
                    f"Skipping tool {tool.name}: {e}"
                )

        return valid_tools

    except Exception as e:
        print("MCP Load Error:", e)
        return []


mcp_tools = load_mcp_tools()

tools = [
    search_tool,
    get_stock_price,
    *mcp_tools
]

print("\n========== LOADED TOOLS ==========")

for t in tools:
    print(
        getattr(t, "name", type(t).__name__)
    )

print("=================================\n")

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# State
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

# ============================================================
# Nodes
# ============================================================


async def chat_node(state: ChatState):

    messages = state["messages"]

    response = await llm_with_tools.ainvoke(
        messages
    )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)

# ============================================================
# Checkpointer
# ============================================================


async def _init_checkpointer():
    conn = await aiosqlite.connect(
        database="chatbot.db"
    )

    return AsyncSqliteSaver(conn)


checkpointer = run_async(
    _init_checkpointer()
)

# ============================================================
# Graph
# ============================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)

chatbot = graph.compile(
    checkpointer=checkpointer
)

# ============================================================
# Threads
# ============================================================


async def _alist_threads():

    threads = set()

    async for checkpoint in checkpointer.alist(None):
        threads.add(
            checkpoint.config[
                "configurable"
            ]["thread_id"]
        )

    return list(threads)


def retrieve_all_threads():
    return run_async(
        _alist_threads()
    )