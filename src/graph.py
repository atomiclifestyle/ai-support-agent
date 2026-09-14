from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.escalation import decide_escalation
from src.intent_classifier import IntentClassifier
from src.judge import judge_reply
from src.rag import RagIndex


class AgentState(TypedDict):
    customer_message: str
    intent: Optional[str]
    intent_confidence: Optional[float]
    should_escalate: Optional[bool]
    escalation_reasons: Optional[List[str]]
    agent_reply: Optional[str]
    retrieved: Optional[list]
    retrieval_similarity: Optional[float]
    judge_scores: Optional[dict]


def build_graph(intent_classifier: IntentClassifier, rag_index: RagIndex):
    def classify_intent_node(state: AgentState):
        intent, confidence = intent_classifier.predict(state["customer_message"])
        return {"intent": intent, "intent_confidence": confidence}

    def escalation_gate_node(state: AgentState):
        _, top_similarity = rag_index.retrieve(state["customer_message"])
        should_escalate, reasons = decide_escalation(
            state["intent"], state["intent_confidence"], top_similarity
        )
        return {
            "should_escalate": should_escalate,
            "escalation_reasons": reasons,
            "retrieval_similarity": top_similarity,
        }

    def route(state: AgentState):
        return "escalate" if state["should_escalate"] else "generate_response"

    def generate_response_node(state: AgentState):
        reply, retrieved, top_similarity = rag_index.generate_reply(state["customer_message"])
        print(reply)
        return {
            "agent_reply": reply,
            "retrieved": retrieved,
            "retrieval_similarity": top_similarity,
        }

    def judge_node(state: AgentState):
        scores = judge_reply(state["customer_message"], state["agent_reply"])
        return {"judge_scores": scores}

    def escalate_node(state: AgentState):
        return {"agent_reply": None}

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("escalation_gate", escalation_gate_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("judge_eval", judge_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "escalation_gate")
    graph.add_conditional_edges(
        "escalation_gate",
        route,
        {"escalate": "escalate", "generate_response": "generate_response"},
    )
    graph.add_edge("generate_response", "judge_eval")
    graph.add_edge("judge_eval", END)
    graph.add_edge("escalate", END)

    return graph.compile()
