import os
import json
from typing import Optional, Any, Dict, List
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import google.generativeai as genai

from langchain_community.utilities.sql_database import SQLDatabase
try:
    # Preferred name in newer versions
    from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool as _QuerySQLTool
except Exception:
    # Backward compatibility
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool as _QuerySQLTool
import uvicorn

# Load environment variables from the .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

POSTGRES_URI = os.getenv("POSTGRES_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not POSTGRES_URI:
    raise RuntimeError("POSTGRES_URI is not set in environment (.env)")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in environment (.env)")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="DBChat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# --- Database setup ---
engine = create_engine(POSTGRES_URI, pool_pre_ping=True)
db = SQLDatabase.from_uri(POSTGRES_URI)
query_tool = _QuerySQLTool(db=db)

def _get_embedding(text: str) -> List[float]:
    """Get embedding from Gemini model."""
    try:
        # Generate embedding using the text-embedding-001 model
        result = genai.embed_content(
            model="embedding-001",
            content=text,
            task_type="retrieval_document",  # or "retrieval_query" for queries
        )
        # Get the embedding values (automatically normalized)
        values = result["embedding"]
        # Ensure we have exactly 3072 dimensions
        if len(values) < 3072:
            values.extend([0.0] * (3072 - len(values)))
        elif len(values) > 3072:
            values = values[:3072]
        return values
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

class KBItem(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    id: int
    text: str
    sender: str

class ChatQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None

@app.post("/kb/add")
def add_to_knowledgebase(item: KBItem) -> Dict[str, Any]:
    try:
        # Generate embedding for the content
        embedding = _get_embedding(item.content)
        
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO knowledge_base (content, metadata, embedding)
                    VALUES (:content, CAST(:metadata AS JSONB), :embedding)
                    RETURNING id
                    """
                ),
                {
                    "content": item.content,
                    "metadata": json.dumps(item.metadata) if item.metadata is not None else None,
                    "embedding": embedding,
                },
            )
            new_id = result.scalar_one()
        return {"id": int(new_id), "status": "ok"}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert knowledge: {str(e)}")

@app.get("/kb/entries")
def get_all_kb_entries() -> Dict[str, Any]:
    """Get all knowledge base entries."""
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT id, content, metadata, created_at
                    FROM knowledge_base
                    ORDER BY created_at DESC
                    """
                )
            ).fetchall()
            
        entries = []
        for row in result:
            entries.append({
                "id": row.id,
                "content": row.content,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat() if row.created_at else None
            })
            
        return {
            "entries": entries,
            "count": len(entries),
            "status": "ok"
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge entries: {str(e)}")

def _extract_text_content(response) -> str:
    """Extract text from Gemini response."""
    if response is None:
        return ""
    try:
        return response.text
    except:
        return str(response)

def _strip_sql_fences(sql_text: str) -> str:
    cleaned = sql_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    # Remove potential leading labels like sql
    cleaned = cleaned.replace("sql\n", "\n").replace("SQL\n", "\n")
    # Extract only the first statement
    if "\n```" in cleaned:
        cleaned = cleaned.split("\n```")[0]
    return cleaned.strip().rstrip(";")

@app.post("/chat/query")
def process_chat_query(body: ChatQuery) -> Dict[str, Any]:
    user_query = body.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get embedding for the query
    try:
        query_embedding = _get_embedding(user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")

    # Find relevant knowledge using vector similarity
    try:
        with engine.begin() as conn:
            similar_docs = conn.execute(
                text(
                    """
                    SELECT content, metadata, 1 - (embedding <=> :query_embedding) as similarity
                    FROM knowledge_base
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT 5
                    """
                ),
                {"query_embedding": f"[{','.join(map(str, query_embedding))}]"}
            ).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search similar documents: {str(e)}")

    # Check if we have relevant knowledge (similarity threshold)
    SIMILARITY_THRESHOLD = 0.8  # Adjust this value as needed
    relevant_docs = [doc for doc in similar_docs if doc.similarity >= SIMILARITY_THRESHOLD]
    
    if not relevant_docs:
        # No relevant knowledge found - return default message
        return {
            "sql": None,
            "results": None,
            "answer": "I don't have specific information about that topic in my knowledge base. Please ask questions related to the available data or add relevant information to the knowledge base first.",
            "similar_docs": [],
            "has_relevant_knowledge": False
        }

    # Format context from similar documents
    context = "\n\n".join(
        f"Document (similarity: {doc.similarity:.2f}):\n{doc.content}"
        for doc in relevant_docs
    )
    
    # Format chat history if provided
    history_context = ""
    if body.history:
        history_context = "\n\nPrevious conversation context:\n"
        # Only include recent history (last 10 messages) to avoid token limits
        recent_history = body.history[-10:] if len(body.history) > 10 else body.history
        for msg in recent_history:
            history_context += f"{msg.sender}: {msg.text}\n"

    # Provide schema context to the LLM
    try:
        schema_ddl = db.get_table_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load schema: {str(e)}")

    sql_prompt = (
        "You are a helpful assistant that writes precise PostgreSQL SQL queries.\n"
        "Given the following database schema, write ONE SELECT query that best answers the user's question.\n"
        "Consider the following relevant context from similar documents:\n\n"
        f"{context}\n\n"
        f"{history_context}\n\n"
        "- Only output the SQL.\n"
        "- Do not include explanations or code fences.\n"
        "- Use valid PostgreSQL syntax and correct table/column names.\n"
        "- Limit results to 50 rows unless the question asks otherwise.\n\n"
        f"SCHEMA:\n{schema_ddl}\n\n"
        f"QUESTION: {user_query}\n"
    )

    try:
        sql_response = model.generate_content(sql_prompt)
        raw_sql = _extract_text_content(sql_response)
        sql = _strip_sql_fences(raw_sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

    if not sql.lower().lstrip().startswith("select"):
        raise HTTPException(status_code=400, detail="Generated SQL is not a SELECT query; refusing to execute.")

    try:
        # Execute via LangChain tool
        result_text = query_tool.run(sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute SQL: {str(e)}")

    # Ask LLM to summarize/answer using the result rows
    try:
        answer_prompt = (
            "You are a helpful analyst. Given the user's question and SQL query results, "
            "write a concise, accurate answer. If the results are tabular, summarize key findings.\n"
            "Consider the conversation history to provide contextually relevant responses.\n\n"
            f"QUESTION: {user_query}\n\n"
            f"{history_context}\n\n"
            f"SQL: {sql}\n\n"
            f"RESULTS:\n{result_text[:6000]}\n"
        )
        answer_response = model.generate_content(answer_prompt)
        answer_text = _extract_text_content(answer_response).strip()
    except Exception:
        # Fall back to returning raw results only
        answer_text = None

    return {
        "sql": sql,
        "results": result_text,
        "answer": answer_text,
        "similar_docs": [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "similarity": doc.similarity
            }
            for doc in relevant_docs
        ],
        "has_relevant_knowledge": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)