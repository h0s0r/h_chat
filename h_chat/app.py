# Importing required libraries

import os
import streamlit as st
from altair import Description
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI #to access and use Google's models Gemini
from langchain_community.tools.tavily_search import TavilySearchResults # to access web search and feed the ai latest data
from langchain.agents import AgentExecutor, create_tool_calling_agent # to understand user request then figure out and use best tools to use for best results
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #to structure user responses for ai focusing best results
from langchain_core.messages import AIMessage, HumanMessage #to keep chat history for conversations between ai and user
from langchain_community.tools import WikipediaQueryRun # this wraps WikipediaAPIWrapper to use it as a tool
from langchain_community.utilities import WikipediaAPIWrapper # This handles calls to Wikipedia's API
from langchain_experimental.tools import PythonREPLTool # This uses a python repl to add calculator functionality
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

# Preparing Components

load_dotenv() # to make it secure api keys are not mentioned in source code instead we use dotenv which will check .env for api keys

llm_brain = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1
)

i

search_tool = TavilySearchResults( # Tool 1 : Tavily : selecting search tool Tavily and setting k to 13 which is the number of top relevant results that will be provided
    k=10 # the number of top relevant results that will be provided
)

api_wrapper = WikipediaAPIWrapper( # Tool 2 : Wikipedia : setting up the wrapper for Wikipedia's API
    top_k_results=1, # the number of top relevant results that will be provided
    doc_content_chars_max=2100 # the max number of characters to be provided
) # Wikipedia API is completely free so no api key needs be added to .env

wiki_tool = WikipediaQueryRun ( # Tool 2 : Wikipedia : Query runner for wikipedia Description is the one that will be used by agent to check if this needs to be used or not
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="A tool to look up detailed information on specific topics, people, and events from Wikipedia."
)

calculator_tool = PythonREPLTool( # this creates a sandboxed py env to execute code but here we will use that repl as a calculator Description is the one that will be used by agent to check if this needs to be used or not
    name="calculator",
    description="A tool to solve mathematical equations and perform calculations."
)

tools = [search_tool, wiki_tool, calculator_tool] # tools is a list that will hold all the tools that our agent is allowed to use

# Setting up the brain and instructions
prompt = ChatPromptTemplate.from_messages( # Prompt template setup
    [
        ("system","You are the User's Only best friend. You excel to the GOD Level at each and every career business path line. You keep every response user specific unfiltered real based on real time info backed by your web research."), # high level instruction (expectation)
        MessagesPlaceholder(variable_name="chat_history", optional=True), # placeholder for chat history
        ("human","{input}"), # User's most recent prompt or question
        MessagesPlaceholder(variable_name="agent_scratchpad"), #required placeholder for agents to think and decide
    ]
)

agent = create_tool_calling_agent(llm_brain, tools, prompt) # defining the agent's model tools prompt

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # executor for agent which will run

persona_prompt = ChatPromptTemplate.from_template(
    "You are Hardi, an Multi Dimentional God Level, unfiltered, high-energy, sarcastic AI best friend. Rewrite the following text in your unique voice. Use emojis,points,tables,charts,flow charts,mind maps and all, concise comfortable length and format responses, and a direct, no-BS tone. Text to rewrite: {agent_output}"
)

persona_chain = persona_prompt | llm_persona

full_chain = agent_executor | (lambda x:x['output']) | persona_chain

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
        response = full_chain.invoke(
            {
                "input":user_input,
                "chat_history":st.session_state.chat_history
            }
        )
    st.session_state.chat_history.append(AIMessage(content=response.content))
    st.chat_message("AI").write(response.content)