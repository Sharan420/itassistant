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
    query_type: str = None
    kb_context: str = None
    original_query: str = None

class QueryClassification(PydanticBaseModel):
    classification: Literal["tool_available", "not_available"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    convert_system_message_to_human=True,
    temperature=0
)
tools = [password_reset_tool, knowledge_base_retriever]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def classify_query_node(state: AgentState):
    """Classify user query as actionable or informational."""
    # Get the original user query
    original_query = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    
    if not original_query:
        return {"query_type": "informational", "original_query": ""}
    
    # Deterministic classification for common actionable patterns
    query_lower = original_query.lower()
    password_keywords = ["password", "reset", "forgot", "change", "unlock", "account"]
    
    # Check for password-related requests (tool available)
    if any(keyword in query_lower for keyword in ["reset", "forgot"]) and "password" in query_lower:
        print(f"Query Classification: tool_available (password reset tool)")
        return {
            "query_type": "tool_available",
            "original_query": original_query
        }
    
    # Check for other tool-available patterns
    # Currently we only have password_reset_tool, so only password-related queries have tools
    # Future: Add more patterns when more tools are available
    
    # Fallback to LLM classification
    classification_prompt = f"""Classify this user query as either 'tool_available' or 'not_available':

tool_available: A specific tool can handle this request (currently only password reset)
informational: No specific tool available, needs knowledge base or general response

Available Tools:
- password_reset_tool: For resetting user passwords

Examples:
- "Reset my password" → tool_available
- "Please reset my password" → tool_available
- "I forgot my password" → tool_available
- "Change my password" → not_available (no tool for password changes)
- "Unlock my account" → not_available (no account unlock tool)
- "How do I connect to VPN?" → not_available
- "What is the WiFi password?" → not_available
- "Hello" → not_available
- "How to reset password?" → not_available (asking how, not requesting action)

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
    query_type = state.get("query_type", "not_available")
    if query_type == "tool_available":
        return "tool_flow"
    else:
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
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=original_query)]
    
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
    """Search knowledge base for informational queries."""
    original_query = state.get("original_query", "")
    
    if not original_query:
        return {"kb_context": ""}
    
    # Search knowledge base
    try:
        kb_result = knowledge_base_retriever.invoke({"query": original_query})
        return {"kb_context": kb_result}
    except Exception as e:
        return {"kb_context": f"Knowledge base error: {str(e)}"}

def kb_router(state: AgentState):
    """Route based on KB search results."""
    kb_context = state.get("kb_context", "")
    
    # Check if KB found relevant results
    if ("No relevant knowledge base entries were found" in kb_context or 
        "Knowledge base search error" in kb_context or
        "Knowledge base is not configured" in kb_context or
        not kb_context or kb_context.strip() == ""):
        print("KB Router: No relevant KB results, sending fallback message")
        return "no_knowledge_response"
    else:
        print("KB Router: Found KB results, generating LLM response with context")
        return "kb_llm_response"

def no_knowledge_response_node(state: AgentState):
    """Return fallback message when no knowledge is available."""
    response_msg = AIMessage(content="I cannot help with that query please contact an Admin")
    return {"messages": [response_msg]}

def kb_llm_response_node(state: AgentState):
    """Generate LLM response using KB context."""
    original_query = state.get("original_query", "")
    kb_context = state.get("kb_context", "")
    
    system_prompt = (
        "Do not mention sources. Only straight resolutions. Do not use markdown syntax for replies."
        "You are an IT assistant. Use the following knowledge base information "
        "to answer the user's question. Be helpful and provide a clear response "
        "based on the available information.\n\n"
        f"Knowledge Base Context:\n{kb_context}"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=original_query)]
    response = llm.invoke(messages)  # Use regular LLM without tools for informational responses
    return {"messages": [response]}

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
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=original_query)]
    
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
graph_builder.add_node("call_tools", tool_node)

# Entry point: classify the query first
graph_builder.set_entry_point("classify_query")

# Route based on classification
graph_builder.add_conditional_edges(
    "classify_query",
    classification_router,
    {
        "tool_flow": "tool_agent",
        "kb_flow": "kb_search"
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

# Informational flow: route based on KB results
graph_builder.add_conditional_edges(
    "kb_search",
    kb_router,
    {
        "kb_llm_response": "kb_llm_response",
        "no_knowledge_response": "no_knowledge_response"
    }
)

# Both KB responses end the flow
graph_builder.add_edge("kb_llm_response", END)
graph_builder.add_edge("no_knowledge_response", END)

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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)