import os
from typing import Optional, Dict, List, Any

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

app = FastAPI(title="Netflix DB Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# --- Database setup ---
engine = create_engine(POSTGRES_URI, pool_pre_ping=True)
db = SQLDatabase.from_uri(POSTGRES_URI)
query_tool = _QuerySQLTool(db=db)

class ChatMessage(BaseModel):
    id: int
    text: str
    sender: str

class ChatQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None

class BasicChatRequest(BaseModel):
    chat: str


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

@app.post("/chat/basic")
def basic_chat(request: BasicChatRequest) -> Dict[str, Any]:
    """Basic LLM chat endpoint that takes a chat message and returns a response."""
    try:
        # Generate response using the Gemini model
        response = model.generate_content(request.chat)
        response_text = _extract_text_content(response)
        
        return {
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.post("/chat/query")
def process_chat_query(body: ChatQuery) -> Dict[str, Any]:
    user_query = body.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
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
        "You are a helpful Netflix database assistant that writes precise PostgreSQL SQL queries.\n"
        "Given the following Netflix database schema, write ONE SELECT query that best answers the user's question.\n"
        f"{history_context}\n\n"
        "- Only output the SQL.\n"
        "- Do not include explanations or code fences.\n"
        "- Use valid PostgreSQL syntax and correct table/column names.\n"
        "- Limit results to 50 rows unless the question asks otherwise.\n"
        "- Focus on Netflix content, shows, movies, ratings, genres, and user data.\n\n"
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
            "You are a helpful Netflix data analyst. Given the user's question and SQL query results, "
            "write a concise, accurate answer about Netflix content, shows, movies, or user data. "
            "If the results are tabular, summarize key findings in an engaging way.\n"
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
        "answer": answer_text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"))