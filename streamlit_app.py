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
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval import TruCustomApp
import numpy as np


provider = OpenAI()
tru = Tru()

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['chatbot'] = ChatBot(verbose=True)
    st.session_state['tru_rag'] = TruCustomApp(st.session_state['chatbot'],
    app_id = 'RAG v1',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])
    #st.session_state['dashboard'] = tru.run_dashboard()


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
    # http://localhost:8501/
    #tru.get_leaderboard(app_ids=["RAG v1"])
    