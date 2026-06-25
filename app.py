import os
import re
import json
import uuid
import asyncio
import logging
import hashlib
import tempfile
import mimetypes
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Form, Cookie
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

from supabase import create_client

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeloXAi")

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# File handling config
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_TEXT_LENGTH = 100000

# Auth config
SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

app = FastAPI(
    title="HeloxAi Lite",
    description="Text, Code, Math, and Research Backend",
    version="3.1.0"
)

# =========================
# CORS CONFIGURATION (FIXED)
# =========================
# Automatically detect the deployed URL to avoid CORS errors
service_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SERVICE_URL") or "https://heloxai2.onrender.com"
frontend_url = os.getenv("FRONTEND_URL", service_url)

allowed_origins = [
    frontend_url,
    service_url, # Allow the backend itself
    "https://heloxai.xyz",
    "https://www.heloxai.xyz",
    "capacitor://localhost", # Mobile apps
]

logger.info(f"CORS Allowed Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, OPTIONS, HEAD, etc.
    allow_headers=["*"], # Allows all headers
    expose_headers=["*"]
)

# Database Clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
active_streams: Dict[str, asyncio.Task] = {}

# Session cache
_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300
_session_cache_last_cleanup = 0

# =========================
# FILE TYPES
# =========================
class FileCategory(Enum):
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    UNKNOWN = "unknown"

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.java', '.c', '.cpp', '.go', '.rs', '.php', '.rb', '.swift', '.sql', '.json', '.yaml', '.xml'
}

DOCUMENT_EXTENSIONS = {
    '.txt', '.md', '.csv', '.pdf', '.doc', '.docx', '.log'
}

DATA_EXTENSIONS = {
    '.csv', '.json', '.xml', '.yaml'
}

def get_file_category(filename: str) -> FileCategory:
    if not filename: return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS: return FileCategory.CODE
    if ext in DOCUMENT_EXTENSIONS: return FileCategory.DOCUMENT
    if ext in DATA_EXTENSIONS: return FileCategory.DATA
    return FileCategory.UNKNOWN

async def extract_text_safe(content: bytes) -> str:
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            return content.decode(enc, errors='ignore')[:MAX_TEXT_LENGTH]
        except:
            continue
    return "[Binary or unreadable content]"

# =========================
# AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    base = {
        "max_age": SESSION_DURATION if remember else 24 * 60 * 60,
        "httponly": True,
        "secure": True,
        "samesite": "none",
        "path": "/"
    }
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    if cookie_domain: base["domain"] = cookie_domain
    return base

def generate_session_token() -> str:
    import secrets
    return secrets.token_urlsafe(64)

def set_session_cookies(response: Response, user_id: str, token: str, remember: bool = True):
    settings = get_cookie_settings(remember)
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)

def clear_session_cookies(response: Response):
    cookies = [PRIMARY_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE]
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    for c in cookies:
        kwargs = {"key": c, "path": "/", "secure": True, "samesite": "none"}
        if cookie_domain: kwargs["domain"] = cookie_domain
        response.delete_cookie(**kwargs)

def is_session_expired(expiry_str: str) -> bool:
    try:
        return time.time() > int(expiry_str)
    except: return True

async def validate_session_token(user_id: str, token: str) -> bool:
    try:
        if user_id in _session_cache and _session_cache[user_id].get("token") == token:
            return True
        
        result = await asyncio.to_thread(
            supabase.table("user_sessions")
            .select("token")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute
        )
        
        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {"token": token}
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

async def create_user_session(user_id: str, remember: bool = True) -> str:
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_DURATION if remember else 24 * 60 * 60)
    try:
        await asyncio.to_thread(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "token": token,
                "expires_at": expires_at.isoformat(),
                "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute
        )
        _session_cache[user_id] = {"token": token}
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
    return token

# =========================
# SYSTEM PROMPTS
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi, a powerful AI assistant.

**Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Coding:** Expert in writing, debugging, and reviewing code across all languages.
3. **Math:** Capable of solving mathematical problems and equations.
4. **Research:** You have access to real-time web search. Use it for current events or facts.

**Response Style:**
- Use Markdown for structure (headers, bolding, code blocks).
- Be concise but thorough.
- If you use web search, cite the source URL.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi"."""

def get_system_prompt(user_prompt: str) -> str:
    return BASE_SYSTEM_PROMPT

# =========================
# INTENT DETECTION
# =========================
class IntentCategory(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    CONVERSATION = "conversation"

@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float

class AdvancedIntentDetector:
    def __init__(self):
        self.patterns = {
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|make)\s+(code|function|script|program)',
                r'\b implement \s+',
                r'\bhow\s+to\s+code\s+'
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve)\s+(this|my|the)\s+(bug|error)',
                r'\bwhy\s+is\s+(this|it)\s+not\s+working',
                r'\berror\s*:'
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|refactor|improve)\s+(this|my)\s+code',
                r'\b(is\s+this)\s+code\s+(good|clean)'
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|solve|compute)\s+',
                r'\b\d+[\+\-\*\/\^]\d+',
                r'\bintegral|derivative|equation\b'
            ],
            IntentCategory.RESEARCH: [
                r'\b(search|find|look\s+up)\s+(for|about)',
                r'\blatest\s+news|current\s+events',
                r'\bwho\s+is\s+(currently|now)'
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|thanks)',
                r'^(how\s+are\s+you)'
            ]
        }
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.patterns.items()
        }

    def detect(self, text: str) -> Optional[IntentResult]:
        for intent, patterns in self.compiled_patterns.items():
            matches = 0
            for p in patterns:
                if p.search(text):
                    matches += 1
            if matches > 0:
                return IntentResult(intent=intent, confidence=min(0.5 + (matches*0.1), 0.95))
        return IntentResult(intent=IntentCategory.CONVERSATION, confidence=0.5)

_detector = AdvancedIntentDetector()

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True

# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def _execute_supabase_with_retry(query_builder):
    try:
        return await asyncio.to_thread(query_builder.execute)
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise

async def get_user(req: Request, res: Response, remember: bool = True) -> Dict[str, Any]:
    user_id = req.cookies.get(PRIMARY_COOKIE)
    token = req.cookies.get(SESSION_TOKEN_COOKIE)
    expiry = req.cookies.get(SESSION_EXPIRY_COOKIE)
    
    if user_id and token:
        if is_session_expired(expiry or "0"):
            clear_session_cookies(res)
        elif await validate_session_token(user_id, token):
            return {"id": user_id, "session_valid": True}

    new_id = str(uuid.uuid4())
    new_token = await create_user_session(new_id, remember)
    set_session_cookies(res, new_id, new_token, remember)
    return {"id": new_id, "session_valid": True, "memory": ""}

async def save_message(user_id: str, conv_id: str, role: str, content: str):
    data = {
        "id": str(uuid.uuid4()),
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await _execute_supabase_with_retry(supabase.table("messages").insert(data))

async def get_history(conv_id: str, limit: int = 20):
    res = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(limit)
    )
    return [{"role": m["role"], "content": m["content"]} for m in (res.data or [])]

# =========================
# API INTEGRATIONS
# =========================
def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

async def perform_web_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "[Search API Key missing]"
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True
            }
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            context = ""
            if "answer" in data: context += f"Answer: {data['answer']}\n"
            for r in data.get("results", []):
                context += f"- {r['title']}: {r['content']}\n"
            return context
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "[Search failed]"

async def stream_groq_chat(messages: list):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.1-8b-instant", "messages": messages, "stream": True, "max_tokens": 1024}
        ) as resp:
            if resp.status_code != 200:
                raise Exception(f"Groq Error: {resp.status_code}")
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]": return
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content")
                        if delta: yield delta
                    except: pass

# =========================
# ENDPOINTS
# =========================
@app.api_route("/", methods=["GET", "HEAD"]) # Added HEAD for health checks
async def root():
    return {"status": "running", "service": "HeloxAi Lite", "features": ["chat", "code", "math", "web_search"]}

@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")
    body = {}
    
    if "application/json" in content_type:
        body = await req.json()
    elif "multipart/form-data" in content_type:
        form = await req.form()
        body = dict(form)
        
        if "file" in form:
            file: UploadFile = form["file"]
            content_bytes = await file.read()
            text_content = await extract_text_safe(content_bytes)
            file_prefix = f"\n\n[FILE CONTENT: {file.filename}]\n{text_content}\n[END FILE]\n"
            body["prompt"] = body.get("prompt", "") + file_prefix

    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    remember = body.get("remember", True)

    if not prompt:
        raise HTTPException(400, "Prompt required")

    user = await get_user(req, res, remember)
    intent = _detector.detect(prompt)

    needs_search = (intent.intent == IntentCategory.RESEARCH)
    search_keywords = ["latest", "news", "current", "price", "weather", "stock", "who is"]
    if any(kw in prompt.lower() for kw in search_keywords):
        needs_search = True

    # Ensure conversation exists
    conversation_valid = False
    if conv_id:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", conv_id).limit(1)
        )
        if check.data:
            conversation_valid = True
        else:
            logger.warning(f"Conversation ID {conv_id} provided but not found in DB.")

    if not conversation_valid:
        conv_id = str(uuid.uuid4())
        logger.info(f"Creating new conversation: {conv_id}")
        
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": conv_id,
                "user_id": user["id"],
                "title": prompt[:50] if len(prompt) > 50 else prompt,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        )

    await save_message(user["id"], conv_id, "user", prompt)

    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            
            try:
                full_text = ""
                search_context = ""
                
                if needs_search:
                    yield sse({"type": "status", "message": "Searching web..."})
                    search_context = await perform_web_search(prompt)
                    yield sse({"type": "status", "message": "Synthesizing answer..."})

                history = await get_history(conv_id)
                system_prompt = get_system_prompt(prompt)
                
                if search_context:
                    system_prompt += f"\n\nWEB SEARCH RESULTS:\n{search_context}\n\nUse these results to answer."

                messages = [{"role": "system", "content": system_prompt}] + history

                async for token in stream_groq_chat(messages):
                    if task.cancelled(): break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                await save_message(user["id"], conv_id, "assistant", full_text)
                yield sse({"type": "done"})

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    else:
        search_context = ""
        if needs_search: search_context = await perform_web_search(prompt)
        
        history = await get_history(conv_id)
        system_prompt = get_system_prompt(prompt)
        if search_context: system_prompt += f"\n\nWEB SEARCH RESULTS:\n{search_context}"
        
        messages = [{"role": "system", "content": system_prompt}] + history
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 1024}
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            await save_message(user["id"], conv_id, "assistant", reply)
            return {"reply": reply}

@app.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text")
    voice = data.get("voice", "alloy")
    
    allowed_voices = ["alloy", "onyx"]
    if voice not in allowed_voices:
        voice = "alloy"

    if not text: raise HTTPException(400, "text required")
    if not OPENAI_API_KEY: raise HTTPException(500, "Missing OpenAI Key")

    async def stream_audio():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={"model": "tts-1", "voice": voice, "input": text, "response_format": "mp3"}
            ) as response:
                if response.status_code != 200:
                    logger.error(f"TTS Error: {response.status_code}")
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "onyx", "name": "Onyx"}
        ]
    }

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not OPENAI_API_KEY: raise HTTPException(500, "Missing OpenAI Key")
    content = await file.read()
    
    async with httpx.AsyncClient(timeout=30) as client:
        files = {"file": (file.filename, content, file.content_type)}
        data = {"model": "whisper-1"}
        try:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=get_openai_headers(),
                files=files, data=data
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"STT Error: {e}")
            raise HTTPException(500, "Speech to text failed")

# =========================
# UTILITIES
# =========================
@app.get("/chat/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    msgs = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
    )
    return {"messages": msgs.data}

@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user = await get_user(req, res)
    result = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True)
    )
    return {"chats": result.data}

@app.post("/session/logout")
async def logout(req: Request, res: Response):
    clear_session_cookies(res)
    return {"status": "logged_out"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
