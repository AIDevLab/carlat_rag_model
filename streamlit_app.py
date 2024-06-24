"""
This is streamlit app for chatbot. It is used only in development/testing.
"""

import os 
import sys
# Get the absolute path to the main directory (my_project)
main_directory = os.path.abspath(os.path.dirname(__file__))
# Add main directory and its subdirectories to sys.path
sys.path.append(main_directory)
sys.path.append("C:\\Users\\dell\\Desktop\\carlat RAG chatbot\\chatbot.py")

import streamlit as st
from chatbot import ChatBot
import numpy as np



st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['chatbot'] = ChatBot(verbose=True)


for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(placeholder= "How can I help you?")
if prompt:
    import os
    os.system('clear')
    print(prompt)

    st.chat_message("user").write(prompt)
    with  st.session_state['tru_rag'] as recording:
        response, history = st.session_state.chatbot.query(prompt, st.session_state['messages'])

    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.chat_message("assistant").write(response)
    