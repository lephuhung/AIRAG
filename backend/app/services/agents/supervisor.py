from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..llm import get_llm_client


class SupervisorState(BaseModel):
    messages: list = []
    intent: str | None = None
    routing_target: Literal["agent_rag", "agent_write", "direct_answer"] | None = None
    final_answer: str | None = None


_CLASSIFIER_SYSTEM = """You are an intent classifier for a RAG system.
Classify the user's message into one of these intents:

1. **greeting**: Simple greetings, hellos, how are you, good morning, etc.
2. **personal**: Personal questions not related to documents (e.g., "what is your name?", "how old are you?")
3. **search**: General document search queries looking for information
4. **list_docs**: Requests to list available documents in the workspace
5. **summarize**: Requests to summarize a specific document or topic
6. **kg_query**: Questions about entities, relationships, or concepts that might be in a knowledge graph
7. **search_doc_num**: Looking for specific document by number or ID (e.g., "document 5", "doc #23")
8. **search_abbr**: Looking up abbreviations or acronyms

Return ONLY the intent name in lowercase, nothing else."""


def _parse_classifier_output(output: str) -> str:
    output = output.strip().lower()
    valid_intents = [
        "greeting",
        "personal",
        "search",
        "list_docs",
        "summarize",
        "kg_query",
        "search_doc_num",
        "search_abbr",
    ]
    for intent in valid_intents:
        if intent in output:
            return intent
    return "search"


def classify_intent(state: SupervisorState) -> SupervisorState:
    messages = state.messages
    if not messages:
        return state

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content
    else:
        user_input = str(last_message)

    llm = get_llm_client()
    response = llm.invoke(
        [SystemMessage(content=_CLASSIFIER_SYSTEM), HumanMessage(content=user_input)]
    )
    intent = _parse_classifier_output(
        response.content if hasattr(response, "content") else str(response)
    )

    state.intent = intent

    if intent in ["greeting", "personal"]:
        state.routing_target = "direct_answer"
    elif intent in [
        "search",
        "list_docs",
        "summarize",
        "kg_query",
        "search_doc_num",
        "search_abbr",
    ]:
        state.routing_target = "agent_rag"
    else:
        state.routing_target = "agent_rag"

    return state


def direct_answer_node(state: SupervisorState) -> SupervisorState:
    messages = state.messages
    last_message = messages[-1] if messages else None

    if not last_message:
        state.final_answer = "Hello! How can I help you today?"
        return state

    user_input = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    user_input_lower = user_input.lower()

    if any(
        greet in user_input_lower
        for greet in [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
    ):
        state.final_answer = "Hello! How can I help you today?"
    elif any(
        phrase in user_input_lower
        for phrase in ["how are you", "how do you do", "what's up"]
    ):
        state.final_answer = "I'm doing well, thank you for asking! I'm here to help you search and analyze your documents. What would you like to know?"
    elif any(
        phrase in user_input_lower
        for phrase in ["what is your name", "who are you", "what are you"]
    ):
        state.final_answer = "I'm your AI assistant for searching and analyzing documents. I can help you find information, summarize content, query knowledge graphs, and more!"
    else:
        state.final_answer = (
            "I understand you're saying: \""
            + user_input
            + '". How can I assist you with your documents today?'
        )

    return state


def create_supervisor() -> StateGraph:
    graph = StateGraph(SupervisorState)

    graph.add_node("classify", classify_intent)
    graph.add_node("direct_answer", direct_answer_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        lambda state: state.routing_target,
        {
            "agent_rag": "agent_rag",
            "agent_write": "agent_write",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("direct_answer", END)

    from .agent_rag import create_agent_rag
    from .agent_write import create_agent_write

    agent_rag_graph = create_agent_rag()
    agent_write_graph = create_agent_write()

    graph.add_node("agent_rag", agent_rag_graph)
    graph.add_node("agent_write", agent_write_graph)

    graph.add_edge("agent_rag", END)
    graph.add_edge("agent_write", END)

    return graph.compile()
