import streamlit as st
import rag as rag
import pandas as pd
import Evaluation_metric as eval
from langchain.llms import OpenAI

st.title('Car Insurance Policy Bot')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

input_file = "policy-booklet-0923.pdf"

def generate_response(query):
    retriever = rag.load_data(input_file)
    # llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.info(rag.rag_answer(retriever,query))
    # st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'How may I help you with your today?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)