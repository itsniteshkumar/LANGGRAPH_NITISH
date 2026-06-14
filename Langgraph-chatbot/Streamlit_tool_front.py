#streamlit_frontend_db.py
import streamlit as st
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from LG_tool_backend import chatbot, retrieve_all_threads


# ======================================================
# Utility Functions
# ======================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():

    new_thread = generate_thread_id()

    st.session_state.thread_id = new_thread

    add_thread(new_thread)

    st.session_state.message_history = []


def load_conversation(thread_id):

    try:

        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        if not state:
            return []

        return state.values.get("messages", [])

    except Exception as e:

        st.error(f"Error loading conversation: {e}")
        return []


# ======================================================
# Session State Setup
# ======================================================

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:

    threads = retrieve_all_threads()

    st.session_state.chat_threads = threads

add_thread(st.session_state.thread_id)

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in reversed(st.session_state.chat_threads):

    if st.sidebar.button(
        thread_id,
        key=f"thread_{thread_id}"
    ):

        st.session_state.thread_id = thread_id

        messages = load_conversation(thread_id)

        history = []

        for msg in messages:

            role = (
                "user"
                if isinstance(msg, HumanMessage)
                else "assistant"
            )

            history.append(
                {
                    "role": role,
                    "content": msg.content
                }
            )

        st.session_state.message_history = history

        st.rerun()

# ======================================================
# Main Chat UI
# ======================================================

st.title("LangGraph + Gemini Chatbot")

for message in st.session_state.message_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ======================================================
# Chat Input
# ======================================================

user_input = st.chat_input(
    "Type your message..."
)

if user_input:

    # Display User Message
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        },
        "metadata": {
            "thread_id": st.session_state.thread_id
        },
        "run_name": "chat_turn",
    }

    # Display Assistant Message with Streaming
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        ai_message = st.write_stream(ai_only_stream())

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )