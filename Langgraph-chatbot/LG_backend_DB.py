from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    #Take user query from state
    messages = state['messages']

    #send to llm
    response = model.invoke(messages)

    #response store state
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

#add_node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

#test
# CONFIG = {'configurable': {'thread_id': 'thread-2'}}

# response = chatbot.invoke(
#     {'messages': [HumanMessage(content='what is my name?')]},
#     config = CONFIG
# )

# print(response)

# message will store inside the thread for all the chat.