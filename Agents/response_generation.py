from .state import AgentState
from langchain_core.messages import AIMessage
from chains.rag_chain import *

def generate_answer(state: AgentState) -> AgentState:
    """
    Generates an answer using RAG based on the chat history, context, and the enhanced query.
    Appends the answer as an AIMessage to the state.
    """
    print("Entering generate_answer")

    if "messages" not in state or not state["messages"]:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state.get("documents", [])
    rephrased_query = state.get("enhanced_query", "")

    response = rag_chain.invoke({
        "history": history,
        "context": documents,
        "question": rephrased_query
    })

    generation = response.content.strip()
    state["messages"].append(AIMessage(content=generation))

    print(f"HealthGuru: {generation}")
    return state


def off_topic_response(state: AgentState) -> AgentState:
    """
    Handles off-topic queries by returning a polite rejection message.
    """
    print("Entering off_topic_response")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    state["messages"].append(
        AIMessage(content="I'm sorry! I am a health assistant. Please ask related to health topics.")
    )
    return state


def greeting_response(state: AgentState) -> AgentState:
    """
    Handles greetings by setting proceed_to_generate to True, but returns an empty document list.
    """
    print("Entering greeting_response")
    state['proceed_to_generate'] = True
    state["documents"] = []
    return state
