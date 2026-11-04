from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.2
)

class WokflowState(TypedDict):
    user_input : str
    ai_message : str

class ChatRequest(BaseModel):
    typed_message : str


def chatbot_func(state : WokflowState):
    prompt = f"""" your are a high quailty Chat Assisatnce give the response to the user message {state["user_input"]}"""
    response = llm.invoke(prompt).content
    return {
        "ai_message" : response
    }


graph = StateGraph(WokflowState)
graph.add_node("chat_bot", chatbot_func)

graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

workflow = graph.compile()

@app.post("/chat")
def Chat_Response(request : ChatRequest):
    initial_state = {
        "user_input" : request.typed_message
    }
    response = workflow.invoke(initial_state)
    return response