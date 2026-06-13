#streamlit_frontend_db.py
import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from LG_backend_DB_gemini import (
    chatbot,
    retrive_all_thread
)

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

    threads = retrive_all_thread()

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

        try:

            ai_message = ""

            placeholder = st.empty()

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):
                if hasattr(chunk, "content") and chunk.content:
                    ai_message += chunk.content
                    placeholder.markdown(ai_message + "▌")

            # Final render without cursor
            placeholder.markdown(ai_message)

        except Exception as e:

            ai_message = f"Error: {e}"

            st.error(ai_message)

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )