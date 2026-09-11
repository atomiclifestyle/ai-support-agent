from typing import TypedDict
from langgraph.graph import START,END, StateGraph

class State(TypedDict):
    message: str

def say(state: State):
    print("Node 1 reached")

    return {
        "message": f'Hello {state["message"]}!'
    }

builder=StateGraph(State)

#Graph building
builder.add_node("hello", say)
builder.add_edge(START, "hello")
builder.add_edge("hello",END)

graph=builder.compile()

result=graph.invoke({
    "message": "LangGraph"
})

print(result)