from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os
import uuid

load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    streaming=True,
    api_key=os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY"),
)

class WokflowState(TypedDict):
    messages: Annotated[list, add_messages]

class ChatRequest(BaseModel):
    human_message: str
    thread_id: Optional[str] = None

def generate_thread_id():
    return str(uuid.uuid4())

async def chatbot_func(state: WokflowState):
    response = await llm.ainvoke(state["messages"])
    return {
        "messages": [response]
    }

graph = StateGraph(WokflowState)
graph.add_node("chat_bot", chatbot_func)

graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

@app.post("/chat")
async def Chat_Response(request: ChatRequest):
    thread_id = request.thread_id or generate_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    async def generate_chunk():
        async for chunk, metadata in workflow.astream(
            {"messages": [HumanMessage(content=request.human_message)]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            if node == "chat_bot" and getattr(chunk, "content", None):
                yield chunk.content

    return StreamingResponse(generate_chunk(), media_type="text/plain")

@app.get("/history/{thread_id}")
async def thread_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = workflow.get_state(config)
    values = (state.values or {}) if state else {}
    messages = values.get("messages", []) or []
    out = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        out.append({"role": role, "content": getattr(msg, "content", "")})
    return {"messages": out}
