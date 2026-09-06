from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.graph import build_graph


load_dotenv()

st.set_page_config(page_title="Weather Advisory Support Bot")
st.title("Weather Advisory Support Bot")
st.caption("Live Open-Meteo weather + external SOP matching. Open-Meteo needs no API key.")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.memory = {}
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about an outdoor activity, for example: Is it safe to cycle in Bhopal today?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking live weather and SOPs..."):
            result = st.session_state.graph.invoke({"user_message": prompt, "memory": st.session_state.memory})
            st.session_state.memory = result.get("memory", st.session_state.memory)
            response = result["final_response"]
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
