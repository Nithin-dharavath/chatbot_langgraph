from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import uuid
import sqlite3

load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    streaming=True,
)

class WokflowState(TypedDict):
    messages: Annotated[list, add_messages]

class ChatRequest(BaseModel):
    human_message: str
    thread_id: Optional[str] = None

def generate_thread_id():
    return str(uuid.uuid4())

def chatbot_func(state: WokflowState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }


def retrive_all_threads():
    all_threads = set()

    for check in checkpointer.list(None):
        each_thread = check.config["configurable"]["thread_id"]
        all_threads.add(each_thread)

    return list(all_threads)

sqlite_connection = sqlite3.connect(database = "chatbot.db", check_same_thread = False)

graph = StateGraph(WokflowState)
graph.add_node("chat_bot", chatbot_func)

graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

checkpointer = SqliteSaver(conn = sqlite_connection)

workflow = graph.compile(checkpointer=checkpointer)

@app.post("/chat")
def Chat_Response(request: ChatRequest):
    thread_id = request.thread_id or generate_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    def generate_chunk():
        for chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=request.human_message)]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            if node == "chat_bot" and getattr(chunk, "content", None):
                yield chunk.content

    return StreamingResponse(generate_chunk(), media_type="text/plain")

@app.get("/history/{thread_id}")
def thread_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = workflow.get_state(config)
    values = (state.values or {}) if state else {}
    messages = values.get("messages", []) or []
    out = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        out.append({"role": role, "content": getattr(msg, "content", "")})
    return {"messages": out}
