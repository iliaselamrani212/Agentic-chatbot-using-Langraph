from dotenv import load_dotenv
load_dotenv()


from langgraph.graph import StateGraph,END,START
from typing import TypedDict ,Annotated
from langchain_core.messages import HumanMessage , BaseMessage , SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings   # [GEMINI] + embeddings
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool 
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langgraph.types import interrupt , Command



import requests
import math 

import sqlite3
load_dotenv()
conn = sqlite3.connect(database = 'chatbot.db' , check_same_thread=False)
checkpointer = SqliteSaver(conn = conn )
# [GEMINI] modèles gratuits (free tier) — clé lue dans .env : GOOGLE_API_KEY
GEMINI_CHAT_MODEL = "gemini-3.5-flash"
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
DB_PATH = "faiss_db_gemini"   # [GEMINI] nouvel index : les vecteurs Ollama ne sont pas compatibles
def ingest_rag_document(file_path):
    documents = PyPDFLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    if os.path.exists(DB_PATH):   
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(chunks)
    else:                          
        vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    vector_store = FAISS.load_local(
        folder_path = DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization = True 
    )
    retriever =vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k':4}
    )
    return retriever
@tool 
def purchase(symbol :str , quantity :str)->dict :
    """
    Simulate purchasing a given quantity of a stock symbol.

HUMAN-IN-THE-LOOP:
Before confirming the purchase, this tool will interrupt
and wait for a human decision ("yes" / anything else)
    """
    decision = interrupt(f"Approve buying {quantity} shares fo {symbol}?(Yes/no)")
    if isinstance(decision,str) and decision.lower() == "yes":

        return {
            "status" :"success",
            "message" : f"Purchase order placed for {quantity} shares of {symbol}",
            "symbol":symbol,
            "quantity":quantity,
        }
    else :
        return{
            "status":"cancelled",
            "message":f"Purchase of {quantity} shares of {symbol} was declined by human",
            "symbol":symbol,
            "quantity":quantity,
        }


@tool 
def rag_tool(query:str)->str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)
    if not documents :
        return "No relevant information was found in the PDF"
    fromatted_documents = []
    for index , document in enumerate(documents,start=1):
        source = document.metadata.get('source',"unknown source")
        page = document.metadata.get('page','unknown page')
        fromatted_documents.append(
            f"Document :{index}\n"
            f"Source :{source}\n"
            f"Page :{page}\n"
            f"Content :  {document.page_content}\n"

        )
    return "\n\n".join(fromatted_documents)



search_tool = TavilySearch(
    max_results = 5 ,
    topic="general",
    search_depth = "advanced"
)



@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=HTM6MLWOXVJCK777"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price,rag_tool,purchase]


llm = llm.bind_tools(tools)
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chatbot(state : ChatbotState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

                "Tool usage instructions:\n"
                "- Use `rag_tool` for questions about the uploaded PDF or document. "
                "Always retrieve relevant document content before answering PDF-related questions.\n"
                "- Use `search_tool` for current events, recent information, or information "
                "that requires an internet search.\n"
                "- Use `calculator` for mathematical calculations. Do not calculate complex "
                "expressions manually when the calculator is available.\n"
                "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
                "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

                "Answer general questions directly when no tool is required. "
                "Do not invent information from the uploaded document. "
                "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
                "After receiving a tool result, provide a clear and helpful final answer."
                "- Use `purchase` when the user wants to buy shares of a stock.\n"

        )
    )
    messages = [
        system_message,
        *state["messages"]
    ]
  
    response = llm.invoke(messages)
    return {'messages':[response]}

graph = StateGraph(ChatbotState)
graph.add_node("chatbot",chatbot)
graph.add_node("tools",ToolNode(tools))
graph.add_edge(START,"chatbot")
graph.add_conditional_edges('chatbot',tools_condition)
graph.add_edge("tools","chatbot")

chat = graph.compile(checkpointer = checkpointer)


def retrieve_all_threads():
    all_threads = {}
    for checkpoint in checkpointer.list(None):
        all_threads[checkpoint.config["configurable"]["thread_id"]] = None
    return list(all_threads)