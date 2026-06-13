#LG_backend_DB_gemini.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3
import os
from dotenv import load_dotenv

# ======================================================
# Load Environment Variables
# ======================================================
load_dotenv()

# ======================================================
# Gemini Model
# ======================================================
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

# ======================================================
# State Definition
# ======================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ======================================================
# LangGraph Node
# ======================================================
def chat_node(state: ChatState):
    messages = state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }


# ======================================================
# SQLite Connection
# ======================================================
conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

# ======================================================
# Checkpointer
# ======================================================
checkpointer = SqliteSaver(conn)

# ======================================================
# Build Graph
# ======================================================
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ======================================================
# Compile Graph
# ======================================================
chatbot = graph.compile(
    checkpointer=checkpointer
)

# ======================================================
# Retrieve Thread IDs
# ======================================================
def retrive_all_thread():
    thread_ids = []
    
    try:
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
        )
        
        rows = cursor.fetchall()
        thread_ids = [row[0] for row in rows]
        
    except Exception as e:
        print("Thread retrieval error:", e)
    
    return thread_ids