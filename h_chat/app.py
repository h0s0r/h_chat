# Importing required libraries

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI #to access and use Google's models Gemini
from langchain_community.tools.tavily_search import TavilySearchResults # to access web search and feed the ai latest data
from langchain.agents import AgentExecutor, create_openai_tools_agent # to understand user request then figure out and use best tools to use for best results
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #to structure user responses for ai focusing best results
from langchain_core.messages import AIMessage, HumanMessage #to keep chat history for conversations between ai and user
# For WikipediaAPIWrapper and WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool # For PythonREPLTool

# Preparing Components

load_dotenv() # to make it secure api keys are not mentioned in source code instead we use dotenv which will check .env for api keys

llm = ChatGoogleGenerativeAI( # selecting llm and setting it up like selecting model and temperature
    model="gemini-2.5-flash-lite",
    temperature=1
)

# Initialize the tools the agent can use.
search_tool = TavilySearchResults(k=10)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2100)
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="A tool to look up detailed information on specific topics, people, and events from Wikipedia."
)
calculator_tool = PythonREPLTool(
    name="calculator",
    description="A tool to solve mathematical equations and perform calculations."
)
tools = [search_tool, wiki_tool, calculator_tool]

# Setting up the brain and instructions
prompt = ChatPromptTemplate.from_messages( # Prompt template setup
    [
        ("system","You are the User's Only best friend. You excel to the GOD Level at each and every career business path line. You keep every response user specific unfiltered real based on real time info backed by your web research."), # high level instruction (expectation)
        MessagesPlaceholder(variable_name="chat_history", optional=True), # placeholder for chat history
        ("human","{input}"), # User's most recent prompt or question
        MessagesPlaceholder(variable_name="agent_scratchpad"), #required placeholder for agents to think and decide
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt) # defining the agent's model tools prompt

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # executor for agent which will run

# Streamlit ui web interface building

st.title("H_Chat(Gemini Edition)")
st.write("Ask me anything")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("Human").write(message.content)
    else:
        st.chat_message("AI").write(message.content)

user_input = st.chat_input("Got anything on your mind?")

if user_input:
    st.chat_message("Human").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Thinking"):
        response = agent_executor.invoke(
            {
                "input":user_input,
                "chat_history":st.session_state.chat_history
            }
        )
    st.session_state.chat_history.append(AIMessage(content=response["output"]))
    st.chat_message("AI").write(response["output"])