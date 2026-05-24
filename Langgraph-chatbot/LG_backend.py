from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

model = ChatOllama(model="llama3.2:latest")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    #Take user query from state
    messages = state['messages']

    #send to llm
    response = model.invoke(messages)

    #response store state
    return {'messages': [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

#add_node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)
