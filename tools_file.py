from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from dotenv import load_dotenv

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.2
)

load_dotenv()

@tool
def calculator(a : float, b : float, operation : str) -> dict:
    """Given the two numbers a and b perform arthermetic operations avaliable:
    add, sub, multiply, division"""
    if operation == "addition":
        return {"result" : (a + b)}
    elif operation == "subtraction":
        return {"result" : (a - b)}
    elif operation == "multiply":
        return {"result" : (a*b)}
    elif operation == "division":
        if b == 0:
            return {"error" : "cant divide by the 0"}
        return {"result" : (a / b)}
    return {"error" : "invalid operation to perform"}

ducksearch = DuckDuckGoSearchRun()


toollist = [calculator, ducksearch]

llm_with_tools = llm.bind_tools(toollist)

tool_node = ToolNode(toollist)


# graph = StateGraph(chatstate)
# graph.add_node("chatnode", chatnode)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chatnode")
# graph.add_conditional_edges("chatnode", tools_condition)
# graph.add_edge("tools", "chatnode")
# workflow = graph.compile()
# workflow