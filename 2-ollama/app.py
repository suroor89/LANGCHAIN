import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

## LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt_template=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant for question answering tasks."),
    ("human","{input}")
])

## Streamlit Framework
st.title("Langchain Demo With Gemma Model")
user_input=st.text_input("What question you have in mind?")

## Ollama LLM
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt_template | llm | output_parser

if user_input:
    response=chain.invoke({
        "input": user_input
    })
    st.write("Response from the model:")
    st.write(response)
