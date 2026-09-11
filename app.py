import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph
from src.intent_classifier import IntentClassifier
from src.rag import RagIndex

st.set_page_config(page_title="AI Support Agent", page_icon="🎧", layout="centered")


@st.cache_resource
def load_pipeline():
    intent_classifier = IntentClassifier()
    rag_index = RagIndex()
    return build_graph(intent_classifier, rag_index)


app = load_pipeline()

st.title("🎧 AI Support Agent")
st.caption("Intent classification -> RAG reply generation -> escalation decision")

message = st.text_area("Customer message", height=110, placeholder="Type a customer tweet...")
run = st.button("Run", type="primary", use_container_width=True)

if run and message.strip():
    with st.spinner("Running pipeline..."):
        result = app.invoke({"customer_message": message})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Intent", result["intent"])
        st.caption(f"confidence: {result['intent_confidence']:.2f}")
    with col2:
        st.metric("Retrieval similarity", f"{result['retrieval_similarity']:.2f}")

    if result["should_escalate"]:
        st.error("Escalated to a human agent")
        for reason in result["escalation_reasons"]:
            st.write(f"- {reason}")
    else:
        st.success("Auto-handled")
        st.subheader("Generated reply")
        st.write(result["agent_reply"])

        with st.expander("Retrieved past examples"):
            for r in result["retrieved"]:
                st.markdown(f"**Customer:** {r['customer_message']}")
                st.markdown(f"**Agent:** {r['agent_reply']}")
                st.divider()

        st.subheader("LLM-as-judge scores")
        st.json(result["judge_scores"])
