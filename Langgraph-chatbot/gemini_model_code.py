from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node
def chat_node(state: ChatState):
    messages = state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }

# Checkpointer
checkpointer = MemorySaver()

# Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

from langchain_core.messages import HumanMessage

CONFIG = {
    "configurable": {
        "thread_id": "stream-test"
    }
}

for message_chunk, metadata in chatbot.stream(
    {
        "messages": [
            HumanMessage(content="Explain Kubernetes in 5 lines")
        ]
    },
    config=CONFIG,
    stream_mode="messages"
):
    if hasattr(message_chunk, "content"):
        print(message_chunk.content, end="", flush=True)