from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.2,
    streaming = True
)

class WokflowState(TypedDict):
    messages : Annotated[list, add_messages]

class ChatRequest(BaseModel):
    human_message : str


def chatbot_func(state : WokflowState):
    response = llm.invoke(state["messages"])
    return {
        "messages" : [response]
    }

graph = StateGraph(WokflowState)
graph.add_node("chat_bot", chatbot_func)

graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

checkpointer = InMemorySaver()
CONFIG = { "configurable" : {
    "thread_id" : 1
}}
workflow = graph.compile(checkpointer = checkpointer)

@app.post("/chat")
async def Chat_Response(request : ChatRequest):

    async def generate_chunk():
        async for chunk, metadata in workflow.astream(
            {"messages" : [HumanMessage(content = request.human_message)]}, config = CONFIG, stream_mode = "messages"):

            if metadata["langgraph_node"] == "chat_bot":
                if chunk.content:
                    yield chunk.content

    return StreamingResponse(generate_chunk(), media_type = "text/plain")
