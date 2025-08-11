import os
from fastapi import FastAPI
from typing import Annotated, List, TypedDict, Literal
from dotenv import load_dotenv
import uvicorn

from pydantic import BaseModel as PydanticBaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
# Import the Google Generative AI chat model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient

# Load environment variables from the .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Mongo / Embeddings lazy singletons
_mongo_client = None
_kb_collection = None
_embeddings = None

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = "itassistant"
MONGODB_COLLECTION = "knowledge"
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "kb_vector_index")

def get_kb_collection():
    global _mongo_client, _kb_collection
    if _kb_collection is not None:
        return _kb_collection
    if not MONGODB_URI:
        return None
    _mongo_client = MongoClient(MONGODB_URI)
    _kb_collection = _mongo_client[MONGODB_DB][MONGODB_COLLECTION]
    return _kb_collection

def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    api_key = os.getenv("GOOGLE_API_KEY")
    _embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    return _embeddings

# --- 1. Define Tools (No changes here) ---
# In a real application, this would make an actual API call
@tool
def password_reset_tool(email: str) -> str:
    """Send a password reset link to the given email address. If no email is provided, ask the user for their email address."""
    print(f"Password reset tool called with email: {email}")
    return f"A password reset link has been sent to {email}."

@tool
def knowledge_base_retriever(query: str) -> str:
    """Retrieve guidance from the IT knowledge base for the provided query using MongoDB Atlas Vector Search."""
    try:
        if not query or not query.strip():
            return "Please provide a query to search the knowledge base."

        collection = get_kb_collection()
        if collection is None:
            return (
                "Knowledge base is not configured. Set MONGODB_URI, MONGODB_DB, "
                "MONGODB_COLLECTION, and VECTOR_INDEX_NAME in the environment."
            )

        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 200,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "source": 1,
                    "title": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(collection.aggregate(pipeline))
        # Debug print of fetched KB results
        try:
            print(f"KB search returned {len(results)} result(s) for query: {query}")
            if results:
                print("KB top document (raw):", results[0])
        except Exception:
            pass
        if not results:
            return "No relevant knowledge base entries were found. Please rephrase your question."

        top = results[0]
        title = top.get("title")
        source = top.get("source")
        score = top.get("score")
        content = top.get("content", "")

        header_parts = []
        if title:
            header_parts.append(f"Title: {title}")
        if source:
            header_parts.append(f"Source: {source}")
        if score is not None:
            header_parts.append(f"Score: {round(float(score), 4)}")
        header = " | ".join(header_parts)

        snippet = content.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."

        if header:
            return f"{header}\n\n{snippet}"
        return snippet
    except Exception as e:
        return f"Knowledge base search error: {str(e)}"

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    query_type: str | None
    kb_context: str | None
    original_query: str | None
    history_used: bool | None

class QueryClassification(PydanticBaseModel):
    classification: Literal["tool_available", "tools_not_available", "tools_not_required"]

class RelevanceDecision(PydanticBaseModel):
    decision: Literal["relevant", "irrelevant"]

class KBAnswer(PydanticBaseModel):
    answer: str

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    convert_system_message_to_human=True,
    temperature=0
)
tools = [password_reset_tool, knowledge_base_retriever]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def classify_query_node(state: AgentState):
    """Classify user query as: tool_available | tools_not_available | tools_not_required."""
    # Get the original user query (most recent human message)
    original_query = None
    last_human_index = None
    for idx, msg in enumerate(reversed(state["messages"])):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            last_human_index = len(state["messages"]) - 1 - idx
            break
    
    if not original_query:
        return {"query_type": "tools_not_required", "original_query": ""}
    
    # Deterministic classification for common actionable patterns
    query_lower = original_query.lower()
    password_keywords = ["password", "reset", "forgot", "change", "unlock", "account"]

    # Helper: did a recent prior user message express password reset intent?
    prior_password_intent = False
    if last_human_index is not None:
        for prior_msg in reversed(state["messages"][:last_human_index]):
            if isinstance(prior_msg, HumanMessage):
                text = str(prior_msg.content).lower()
                if ("password" in text) and ("reset" in text or "forgot" in text):
                    prior_password_intent = True
                    break
    
    # Check for password-related requests (tool available)
    if (any(keyword in query_lower for keyword in ["reset", "forgot"]) and "password" in query_lower) or prior_password_intent:
        print(f"Query Classification: tool_available (password reset tool)")
        return {
            "query_type": "tool_available",
            "original_query": original_query
        }
    
    # Check for other action requests we DO NOT have tools for
    unavailable_action_patterns = [
        ("unlock", "account"),
        ("change", "password"),
        ("disable", "account"),
        ("enable", "account"),
        ("create", "account")
    ]
    for pattern in unavailable_action_patterns:
        if all(word in query_lower for word in pattern):
            print("Query Classification: tools_not_available (no matching tool)")
            return {
                "query_type": "tools_not_available",
                "original_query": original_query
            }

    # Heuristic: greetings / small talk / general statements → tools_not_required
    general_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "test", "testing"
    ]
    if any(p in query_lower for p in general_patterns):
        print("Query Classification: tools_not_required (general query)")
        return {"query_type": "tools_not_required", "original_query": original_query}
    
    # Fallback to LLM classification
    classification_prompt = f"""Classify the user query into exactly one of:

- tool_available: A specific tool can execute this request. Currently only password resets are supported.
- tools_not_available: The user requests an action we cannot execute via tools (e.g., account unlock, password change).
- tools_not_required: The user is asking for information, guidance, or a general message (greetings, small talk, how-to).

Available Tools:
- password_reset_tool: For sending a password reset link given an email.

Examples:
- "Reset my password" → tool_available
- "Please reset my password" → tool_available
- "I forgot my password" → tool_available
- "Change my password" → tools_not_available
- "Unlock my account" → tools_not_available
- "How do I connect to VPN?" → tools_not_required
- "What is the WiFi password?" → tools_not_required
- "Hello" → tools_not_required
- "How to reset password?" → tools_not_required

Query: {original_query}"""
    
    classifier = llm.with_structured_output(QueryClassification)
    result = classifier.invoke(classification_prompt)
    
    print(f"Query Classification: {result.classification}")
    return {
        "query_type": result.classification,
        "original_query": original_query
    }

def classification_router(state: AgentState):
    """Route based on query classification."""
    query_type = state.get("query_type", "tools_not_required")
    if query_type == "tool_available":
        return "tool_flow"
    if query_type == "tools_not_available":
        return "capability_limited"
    return "kb_flow"

def tool_agent_node(state: AgentState):
    """Handle queries that have specific tools available."""
    original_query = state.get("original_query", "")
    
    # For password reset requests, be very explicit
    if any(word in original_query.lower() for word in ["reset", "forgot"]) and "password" in original_query.lower():
        system_prompt = (
            "You are an IT assistant handling a password reset request. "
            "You MUST use the password_reset_tool function. Do not use markdown syntax. "
            "If the user provided an email, call password_reset_tool(email='their_email') immediately. "
            "If no email was provided, ask 'What is your email address?' first, then use the tool. "
            "Do not give any other response. Use the tool."
        )
    else:
        system_prompt = (
            "You are an IT assistant. A specific tool is available for this request. "
            "You MUST use the available tool. Do not give generic responses. "
            "Do not use markdown syntax."
        )
    
    # Include full prior history to let the model infer missing details from context
    history_messages = state.get("messages", [])
    messages = [*history_messages, SystemMessage(content=system_prompt), HumanMessage(content=original_query)]
    
    # Use dedicated LLM instance for tool usage with higher temperature
    tool_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        convert_system_message_to_human=True,
        temperature=0.3  # Even higher temperature to encourage tool usage
    ).bind_tools(tools)
    
    response = tool_llm.invoke(messages)
    
    # Debug: Print what the LLM returned
    print(f"Tool Agent Response: {response}")
    if hasattr(response, 'tool_calls'):
        print(f"Tool Calls: {response.tool_calls}")
    
    return {"messages": [response]}

def kb_search_node(state: AgentState):
    """Search knowledge base for tools_not_required queries."""
    original_query = state.get("original_query", "")
    
    if not original_query:
        return {"kb_context": ""}
    
    # Search knowledge base
    try:
        kb_result = knowledge_base_retriever.invoke({"query": original_query})
        return {"kb_context": kb_result}
    except Exception as e:
        return {"kb_context": f"Knowledge base error: {str(e)}"}

def kb_relevance_router(state: AgentState):
    """Use LLM to judge if KB context is relevant to the query; route accordingly."""
    kb_context = state.get("kb_context", "")
    original_query = state.get("original_query", "")

    # Quick negative checks
    if ("No relevant knowledge base entries were found" in kb_context or 
        "Knowledge base search error" in kb_context or
        "Knowledge base is not configured" in kb_context or
        not kb_context or kb_context.strip() == ""):
        print("KB Relevance: No KB content to evaluate → no_knowledge_response")
        return "no_knowledge_response"

    prompt = (
        "You are a strict relevance judge. Decide if the provided knowledge base content directly answers the user's query. "
        "Return strictly one of: relevant, irrelevant. Be conservative: if unsure, return 'irrelevant'. "
        "Do not use markdown syntax.\n\n"
        f"User Query:\n{original_query}\n\nKB Content:\n{kb_context}"
    )
    relevance_chain = llm.with_structured_output(RelevanceDecision)
    decision = relevance_chain.invoke(prompt)
    print(f"KB Relevance decision: {decision.decision}")
    if decision.decision == "relevant":
        return "kb_llm_response"
    return "no_knowledge_response"

def capability_limited_node(state: AgentState):
    """Respond that tools are not available for this request."""
    return {"messages": [AIMessage(content="Currently Out of my capability")]}

def no_knowledge_response_node(state: AgentState):
    """Return fixed message when no relevant KB is found. Do not generate new content."""
    return {"messages": [AIMessage(content="No Knowledge Stored")]}

def kb_llm_response_node(state: AgentState):
    """Generate LLM response using KB context and chat history, with strict KB-only policy."""
    original_query = state.get("original_query", "")
    kb_context = state.get("kb_context", "")
    history_messages: List[BaseMessage] = state.get("messages", []) or []

    # Compact textual history for additional context without enabling extra knowledge
    history_lines: List[str] = []
    for msg in history_messages[-6:]:
        role = "user" if isinstance(msg, HumanMessage) else ("assistant" if isinstance(msg, AIMessage) else "system")
        history_lines.append(f"{role}: {str(msg.content)}")
    history_text = "\n".join(history_lines)

    prompt = (
        "You are an IT assistant. Do not use markdown syntax.\n"
        "Rewrite an answer STRICTLY using ONLY the provided knowledge base content.\n"
        "- Do NOT add any information that is not present in the KB content.\n"
        "- If the KB content does not contain enough information to answer, output exactly: No Knowledge Stored\n"
        "- Provide a concise resolution-only answer.\n\n"
        f"Conversation history (may contain prior context; do NOT introduce new info beyond KB):\n{history_text}\n\n"
        f"User Query:\n{original_query}\n\n"
        f"Knowledge Base Content:\n{kb_context}\n\n"
        "Return the final answer text."
    )

    chain = llm.with_structured_output(KBAnswer)
    result = chain.invoke(prompt)
    answer_text = result.answer.strip() if hasattr(result, "answer") else str(result).strip()
    if not answer_text:
        answer_text = "No Knowledge Stored"
    return {"messages": [AIMessage(content=answer_text)]}

def tool_router(state: AgentState):
    """Route tool queries - if no tools requested, ask for missing inputs."""
    if not state.get("messages"):
        return "missing_inputs"
    
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print("Tool Router: LLM requested tools")
        return "call_tools"
    else:
        print("Tool Router: No tools requested, asking for missing inputs")
        return "missing_inputs"

def missing_inputs_node(state: AgentState):
    """Handle cases where tools are available but LLM didn't request them - ask for missing inputs."""
    original_query = state.get("original_query", "")
    
    # For password reset, specifically ask for email
    if any(word in original_query.lower() for word in ["reset", "forgot"]) and "password" in original_query.lower():
        system_prompt = (
            "You are an IT assistant. The user wants to reset their password but didn't provide their email address. "
            "Ask them for their email address so you can proceed with the password reset. "
            "Do not use markdown syntax. Be direct and helpful."
        )
    else:
        system_prompt = (
            "You are an IT assistant. The user has made a request that requires additional information. "
            "Ask them for the missing details needed to complete their request. "
            "Do not use markdown syntax. Be direct and helpful."
        )
    
    # Include history for continuity
    history_messages = state.get("messages", [])
    messages = [*history_messages, SystemMessage(content=system_prompt), HumanMessage(content=original_query)]
    
    # Use regular LLM without tools for asking for inputs
    response = llm.invoke(messages)
    return {"messages": [response]}

graph_builder = StateGraph(AgentState)

# Add all nodes
graph_builder.add_node("classify_query", classify_query_node)
graph_builder.add_node("tool_agent", tool_agent_node)
graph_builder.add_node("missing_inputs", missing_inputs_node)
graph_builder.add_node("kb_search", kb_search_node)
graph_builder.add_node("kb_llm_response", kb_llm_response_node)
graph_builder.add_node("no_knowledge_response", no_knowledge_response_node)
graph_builder.add_node("capability_limited", capability_limited_node)
graph_builder.add_node("call_tools", tool_node)

# Entry point: classify the query first
graph_builder.set_entry_point("classify_query")

# Route based on classification
graph_builder.add_conditional_edges(
    "classify_query",
    classification_router,
    {
        "tool_flow": "tool_agent",
        "kb_flow": "kb_search",
        "capability_limited": "capability_limited"
    }
)

# Tool flow: check if tools are needed
graph_builder.add_conditional_edges(
    "tool_agent",
    tool_router,
    {
        "call_tools": "call_tools",
        "missing_inputs": "missing_inputs"
    }
)

# After tools, end
graph_builder.add_edge("call_tools", END)

# Missing inputs response ends the flow
graph_builder.add_edge("missing_inputs", END)

# Informational flow: route based on KB relevance (LLM-judged)
graph_builder.add_conditional_edges(
    "kb_search",
    kb_relevance_router,
    {
        "kb_llm_response": "kb_llm_response",
        "no_knowledge_response": "no_knowledge_response"
    }
)

# Both KB responses end the flow
graph_builder.add_edge("kb_llm_response", END)
graph_builder.add_edge("no_knowledge_response", END)
graph_builder.add_edge("capability_limited", END)

runnable_graph = graph_builder.compile()

#FASTAPI
app = FastAPI(
    title="IT Assistant API with Gemini 1.5 Flash",
    description="An API for interacting with the LangGraph-powered IT Assistant."
)

class QueryRequest(PydanticBaseModel):
    query: str

class QueryResponse(PydanticBaseModel):
    response: str

class AddKnowledgeRequest(PydanticBaseModel):
    content: str
    title: str = None
    source: str = None
    tags: List[str] = []

class AddKnowledgeResponse(PydanticBaseModel):
    success: bool
    message: str
    document_id: str = None

@app.post("/it-assistant", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    system_policy = (
        "You are an IT assistant. Please help the user with their request."
    )
    print(request.query)
    inputs = {"messages": [SystemMessage(content=system_policy), HumanMessage(content=request.query)]}
    final_state = runnable_graph.invoke(inputs, config={"recursion_limit": 8})
    response_message = final_state['messages'][-1].content
    
    return QueryResponse(response=response_message)

@app.post("/knowledge-base/add", response_model=AddKnowledgeResponse)
async def add_knowledge(request: AddKnowledgeRequest):
    """Add a new document to the knowledge base with vector embedding."""
    try:
        if not request.content or not request.content.strip():
            return AddKnowledgeResponse(
                success=False,
                message="Content cannot be empty"
            )
        
        collection = get_kb_collection()
        if collection is None:
            return AddKnowledgeResponse(
                success=False,
                message="Knowledge base is not configured. Set MONGODB_URI and related environment variables."
            )
        
        embeddings = get_embeddings()
        if embeddings is None:
            return AddKnowledgeResponse(
                success=False,
                message="Embeddings service is not configured. Set GOOGLE_API_KEY."
            )
        
        # Generate embedding for the content
        content_embedding = embeddings.embed_query(request.content)
        
        # Prepare document
        document = {
            "content": request.content.strip(),
            "embedding": content_embedding
        }
        
        if request.title:
            document["title"] = request.title.strip()
        if request.source:
            document["source"] = request.source.strip()
        if request.tags:
            document["tags"] = [tag.strip() for tag in request.tags if tag.strip()]
        
        # Insert into MongoDB
        result = collection.insert_one(document)
        
        return AddKnowledgeResponse(
            success=True,
            message="Document successfully added to knowledge base",
            document_id=str(result.inserted_id)
        )
        
    except Exception as e:
        return AddKnowledgeResponse(
            success=False,
            message=f"Error adding document to knowledge base: {str(e)}"
        )

# ==========================
# PSEUDOCODE GRAPH IMPLEMENTATION (v2)
# ==========================

# State for v2 graph
class PState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    route: Literal["tool", "kb"] | None
    slots: dict | None
    missing_fields: List[str] | None
    kb_context: str | None
    relevant: bool | None


def p_context_collector_node(state: PState):
    history_messages: List[BaseMessage] = state.get("messages", []) or []
    # Extract last human query
    user_text = ""
    for msg in reversed(history_messages):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break
    # Simple slot extraction from history (email if present)
    email_value: str | None = None
    for msg in reversed(history_messages):
        if isinstance(msg, HumanMessage):
            text = str(msg.content)
            # Very lightweight pattern capture; LLM will still drive slot decisions
            if "@" in text and "." in text and len(text) <= 128:
                email_value = text.strip()
                break
    slots = state.get("slots") or {}
    if email_value:
        slots["email"] = email_value
    return {
        "messages": [],
        "slots": slots,
    }


class PRoute(PydanticBaseModel):
    route: Literal["tool", "kb"]


def p_intent_classifier_node(state: PState):
    # Decide route using last human message plus recent history context
    history_messages: List[BaseMessage] = state.get("messages", []) or []
    last_human = next((m for m in reversed(history_messages) if isinstance(m, HumanMessage)), None)
    last_text = (last_human.content if last_human else "").lower()

    # Heuristics: password reset → tool, else kb
    tool_intent = ("password" in last_text) and ("reset" in last_text or "forgot" in last_text)
    if not tool_intent:
        # Check earlier human messages for tool intent
        for msg in reversed(history_messages):
            if isinstance(msg, HumanMessage):
                t = str(msg.content).lower()
                if ("password" in t) and ("reset" in t or "forgot" in t):
                    tool_intent = True
                    break

    if tool_intent:
        route = "tool"
    else:
        # Non-tool queries route to KB
        route = "kb"

    # Determine missing fields for known tool intents
    slots = state.get("slots") or {}
    missing_fields: List[str] = []
    if route == "tool":
        if not slots.get("email"):
            missing_fields.append("email")

    return {
        "route": route,
        "missing_fields": missing_fields,
    }


def p_tool_execution_node(state: PState):
    # If missing fields, let router send to slot filler
    missing = state.get("missing_fields") or []
    if missing:
        return {"messages": []}

    # Execute tool using slots
    slots = state.get("slots") or {}
    email = slots.get("email")
    if not email:
        # Safety guard; ask user for email
        ask = AIMessage(content="Please provide your email address to proceed with password reset.")
        return {"messages": [ask], "missing_fields": ["email"]}

    result_text = password_reset_tool.invoke({"email": email}) if hasattr(password_reset_tool, "invoke") else password_reset_tool(email)
    return {"messages": [AIMessage(content=str(result_text))]}


def p_slot_filler_node(state: PState):
    # Ask for any missing fields
    missing = state.get("missing_fields") or []
    if not missing:
        return {"messages": []}
    prompt_parts = ["I need the following information to proceed:"]
    for f in missing:
        if f == "email":
            prompt_parts.append("- Your email address")
        else:
            prompt_parts.append(f"- {f}")
    prompt_parts.append("Please reply with the missing details.")
    content = "\n".join(prompt_parts)
    return {"messages": [AIMessage(content=content)]}


def p_kb_retrieval_node(state: PState):
    history_messages: List[BaseMessage] = state.get("messages", []) or []
    last_human = next((m for m in reversed(history_messages) if isinstance(m, HumanMessage)), None)
    query_text = last_human.content if last_human else ""
    try:
        kb_result = knowledge_base_retriever.invoke({"query": query_text})
    except Exception as e:
        kb_result = f"Knowledge base error: {e}"
    return {"kb_context": kb_result}


class PRelevance(PydanticBaseModel):
    decision: Literal["relevant", "irrelevant"]


def p_relevancy_checker_node(state: PState):
    kb_context = state.get("kb_context") or ""
    history_messages: List[BaseMessage] = state.get("messages", []) or []
    last_human = next((m for m in reversed(history_messages) if isinstance(m, HumanMessage)), None)
    query_text = last_human.content if last_human else ""
    # Quick negatives
    if ("No relevant knowledge base entries were found" in kb_context or
        "Knowledge base search error" in kb_context or
        "Knowledge base is not configured" in kb_context or
        not kb_context.strip()):
        return {"relevant": False}
    prompt = (
        "Return strictly one of: relevant, irrelevant. Do not use markdown syntax.\n\n"
        f"User Query:\n{query_text}\n\nKB Content:\n{kb_context}"
    )
    chain = llm.with_structured_output(PRelevance)
    dec = chain.invoke(prompt)
    return {"relevant": dec.decision == "relevant"}


def p_kb_response_node(state: PState):
    history_messages: List[BaseMessage] = state.get("messages", []) or []
    last_human = next((m for m in reversed(history_messages) if isinstance(m, HumanMessage)), None)
    query_text = last_human.content if last_human else ""
    kb_context = state.get("kb_context") or ""
    sys = (
        "Do not use markdown syntax. You are an IT assistant. Use the provided knowledge context to answer. "
        "Be concise and provide a clear resolution only.\n\n"
        f"Knowledge Context:\n{kb_context}"
    )
    messages = [*history_messages, SystemMessage(content=sys), HumanMessage(content=query_text)]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


# Build v2 graph
p_builder = StateGraph(PState)
p_builder.add_node("context_collector", p_context_collector_node)
p_builder.add_node("intent_classifier", p_intent_classifier_node)
p_builder.add_node("tool_execution", p_tool_execution_node)
p_builder.add_node("slot_filler", p_slot_filler_node)
p_builder.add_node("kb_retrieval", p_kb_retrieval_node)
p_builder.add_node("relevancy_checker", p_relevancy_checker_node)
p_builder.add_node("kb_response", p_kb_response_node)
p_builder.add_node("no_knowledge", lambda s: {"messages": [AIMessage(content="No Knowledge Stored")]})

p_builder.set_entry_point("context_collector")

# context_collector -> intent_classifier
p_builder.add_edge("context_collector", "intent_classifier")

# intent_classifier conditional to tool or kb
def _route_picker(state: PState):
    return state.get("route", "kb")

p_builder.add_conditional_edges(
    "intent_classifier",
    _route_picker,
    {"tool": "tool_execution", "kb": "kb_retrieval"},
)

# Tool side: if missing_fields then slot_filler else END
def _tool_router(state: PState):
    return "slot_filler" if (state.get("missing_fields") or []) else END

p_builder.add_conditional_edges(
    "tool_execution",
    _tool_router,
    {"slot_filler": "slot_filler", END: END},
)
# Slot filler ends (wait for user to supply info in next turn)
p_builder.add_edge("slot_filler", END)

# KB side
p_builder.add_edge("kb_retrieval", "relevancy_checker")

def _relevance_router(state: PState):
    return "kb_response" if state.get("relevant") else "no_knowledge"

p_builder.add_conditional_edges(
    "relevancy_checker",
    _relevance_router,
    {"kb_response": "kb_response", "no_knowledge": "no_knowledge"},
)

p_builder.add_edge("kb_response", END)
p_builder.add_edge("no_knowledge", END)

runnable_graph_v2 = p_builder.compile()


# Optional: second endpoint using v2 graph
@app.post("/it-assistant-v2", response_model=QueryResponse)
async def process_query_v2(request: QueryRequest):
    system_policy = (
        "You are an IT assistant. Do not use markdown syntax. Respond succinctly and factually."
    )
    inputs = {"messages": [SystemMessage(content=system_policy), HumanMessage(content=request.query)]}
    final_state = runnable_graph_v2.invoke(inputs, config={"recursion_limit": 6})
    response_message = final_state["messages"][ -1 ].content
    return QueryResponse(response=response_message)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)