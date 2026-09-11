from typing import TypedDict
from langgraph.graph import START,END, StateGraph
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import load_dataset


class State(TypedDict):
    message: str

def download_dataset(state: State):
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        "thoughtvector/customer-support-on-twitter",
        output_dir=str(data_dir)
    )

    print(f"Dataset downloaded to: {dataset_path}")

    banking77 = load_dataset("mteb/banking77")

    banking77.save_to_disk(data_dir / "banking77")

    print("Banking77 downloaded.")

    return {
        "twitter_dataset_path": str(dataset_path),
        "banking77_path": str(data_dir / "banking77")
    }

def load_dataset():
    return

def intent_classification(state: State):
    print("Node 1 reached")

    return {
        "message": f'Hello {state["message"]}!'
    }

def should_escalate(state: State):
    print("Node 1 reached")

    return {
        "message": f'Hello {state["message"]}!'
    }

def rag_pipeline(state: State):
    print("Node 1 reached")

    return {
        "message": f'Hello {state["message"]}!'
    }

builder=StateGraph(State)

#Graph building
builder.add_node("dataset", download_dataset)
builder.add_edge(START, "dataset")
builder.add_edge("dataset",END)

graph=builder.compile()

result=graph.invoke({
    "message": "LangGraph"
})

# print(result)