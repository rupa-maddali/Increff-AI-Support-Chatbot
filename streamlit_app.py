import streamlit as st
from chatbot import chat_once, memory  # re‑use core pipeline

st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Increff AI Support Chatbot")

if "log" not in st.session_state:
    st.session_state.log = []

def submit():
    user = st.session_state.user_input.strip()
    if user:
        bot = chat_once(user)
        st.session_state.log.append(("You", user))
        st.session_state.log.append(("Bot", bot))
        st.session_state.user_input = ""

st.text_input("Ask me anything about your order or our policies:",
              key="user_input", on_change=submit)

for speaker, msg in st.session_state.log[::-1]:
    st.markdown(f"**{speaker}:** {msg}")

# Clear history button
col1, col2 = st.columns([1,5])
if col1.button("🔄 Reset Chat"):
    st.session_state.log.clear()
    memory.max_turns.clear()
