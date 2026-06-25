import os
import re
import json
import uuid
import asyncio
import logging
import zipfile
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
from io import BytesIO
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path

import httpx
from supabase import create_client
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeloXAi")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

MODEL_NAME = "llama-3.1-8b-instant"
MODEL_DISPLAY = "Llama 8B"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = 0.7
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_TEXT_LENGTH = 50000
SESSION_DURATION = 365 * 24 * 60 * 60
TOKENS_PER_MINUTE_LIMIT = 5500

logger.info(f"Model: {MODEL_NAME}, MaxTokens: {MAX_TOKENS}, Env: {ENVIRONMENT}")
logger.info(f"GROQ_API_KEY set: {bool(GROQ_API_KEY)}, TAVILY set: {bool(TAVILY_API_KEY)}")

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"HeloxAi v4.4.0 started — {MODEL_DISPLAY} via Groq")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="HeloXAi API", version="4.4.0", lifespan=lifespan)

# =========================
# CORS — Must include your frontend domain EXACTLY
# =========================
# Add EVERY domain the frontend runs on (no trailing slash)
FRONTEND_ORIGINS = [
    "https://heloxai.xyz",
    "https://www.heloxai.xyz",
    "https://heloxai2.onrender.com",
    # Add localhost for local dev:
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8080",
]

# Allow extra origins from env
_extra = os.getenv("ALLOWED_ORIGINS", "")
if _extra:
    for o in _extra.split(","):
        o = o.strip()
        if o and o not in FRONTEND_ORIGINS:
            FRONTEND_ORIGINS.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Log the CORS origins so we can verify
logger.info(f"CORS origins: {FRONTEND_ORIGINS}")

# =========================
# SUPABASE (optional)
# =========================
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase initialized")
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")

active_streams: Dict[str, asyncio.Task] = {}

# =========================
# FILE TYPES
# =========================
class FileCategory(Enum):
    CODE="code"; DOCUMENT="document"; DATA="data"; ARCHIVE="archive"; CONFIG="config"; BINARY="binary"; UNKNOWN="unknown"

CODE_EXT = {'.py','.pyw','.js','.jsx','.ts','.tsx','.html','.css','.scss','.vue','.svelte',
            '.java','.kt','.c','.cpp','.cs','.go','.rs','.php','.rb','.swift','.dart',
            '.sh','.bash','.sql','.json','.yaml','.yml','.toml','.md','.dockerfile','.graphql','.tf','.hcl','.sol'}
DOC_EXT = {'.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.txt','.csv','.rtf'}
DATA_EXT = {'.csv','.tsv','.json','.xml','.yaml','.parquet','.pkl','.npy'}
ARCHIVE_EXT = {'.zip','.tar','.gz','.tgz','.bz2','.xz','.7z','.rar'}
CONFIG_EXT = {'.json','.yaml','.yml','.toml','.ini','.cfg','.conf','.env','.xml'}

def get_file_category(fn: str) -> FileCategory:
    if not fn: return FileCategory.UNKNOWN
    ext = Path(fn).suffix.lower()
    if ext in CODE_EXT: return FileCategory.CODE
    if ext in DOC_EXT: return FileCategory.DOCUMENT
    if ext in DATA_EXT: return FileCategory.DATA
    if ext in ARCHIVE_EXT: return FileCategory.ARCHIVE
    if ext in CONFIG_EXT: return FileCategory.CONFIG
    return FileCategory.UNKNOWN

def get_file_language(fn: str) -> Optional[str]:
    m = {'.py':'python','.js':'javascript','.ts':'typescript','.html':'html','.css':'css',
         '.vue':'vue','.java':'java','.cpp':'cpp','.go':'go','.rs':'rust','.sql':'sql',
         '.json':'json','.yaml':'yaml','.md':'markdown','.sh':'bash','.swift':'swift'}
    return m.get(Path(fn).suffix.lower())

def is_binary_file(fn: str, content: bytes = None) -> bool:
    ext = Path(fn).suffix.lower()
    if ext in {'.exe','.dll','.so','.bin','.zip','.tar','.gz','.7z','.rar','.pdf',
                '.doc','.docx','.xls','.xlsx','.png','.jpg','.jpeg','.gif','.webp',
                '.mp3','.mp4','.wav','.avi','.mov','.mkv','.woff','.ttf','.sqlite'}: return True
    if content and len(content) > 0 and b'\x00' in content[:8192]: return True
    return False

def fmt_size(s: int) -> str:
    for u in ['B','KB','MB','GB']:
        if s < 1024.0: return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{s:.1f} TB"

# =========================
# FILE EXTRACTOR
# =========================
class FileResult:
    def __init__(self, content: str, files=None, metadata=None, truncated=False, original_size=0):
        self.content=content; self.files=files or []; self.metadata=metadata or {}
        self.truncated=truncated; self.original_size=original_size
    def to_dict(self):
        return {"content":self.content,"files":self.files,"metadata":self.metadata,
                "truncated":self.truncated,"original_size":self.original_size}

async def extract_file(content: bytes, fn: str, ml=MAX_TEXT_LENGTH) -> FileResult:
    sz=len(content); cat=get_file_category(fn)
    meta={"filename":fn,"category":cat.value,"size":sz,"size_formatted":fmt_size(sz),"language":get_file_language(fn)}
    try:
        if cat==FileCategory.ARCHIVE: return await _extract_arch(content,fn,ml,meta)
        if fn.lower().endswith('.pdf'): return await _extract_pdf(content,fn,ml,meta)
        if is_binary_file(fn,content): return FileResult(f"[Binary: {fn}]",meta,original_size=sz)
        text,trunc=_decode(content,ml); meta["line_count"]=text.count('\n')+1
        return FileResult(text,meta=meta,truncated=trunc,original_size=sz)
    except Exception as e:
        return FileResult(f"[Error: {e}]",{**meta,"error":str(e)},original_size=sz)

def _decode(content,ml):
    for enc in ['utf-8','utf-8-sig','latin-1','cp1252']:
        try:
            t=content.decode(enc,errors='strict' if enc!='latin-1' else 'ignore')
            if len(t)>ml: t=t[:ml]+"\n\n[... truncated ...]"
            return t,len(t)>ml
        except: continue
    t=content.decode('utf-8',errors='replace')
    if len(t)>ml: t=t[:ml]+"\n\n[... truncated ...]"
    return t,len(t)>ml

async def _extract_pdf(content,fn,ml,meta):
    try:
        from PyPDF2 import PdfReader
        pages=[f"--- Page {i+1} ---\n{p.extract_text() or ''}" for i,p in enumerate(PdfReader(BytesIO(content)).pages)]
        txt="\n\n".join(pages); meta["page_count"]=len(pages)
        if len(txt)>ml: txt=txt[:ml]+"\n\n[... truncated ...]"
        return FileResult(txt,meta=meta,truncated=len(txt)>ml,original_size=len(content))
    except ImportError:
        return FileResult(f"[PDF: {fn} - no parser]",meta,original_size=len(content))

async def _extract_arch(content,fn,ml,meta):
    ext=Path(fn).suffix.lower()
    if ext=='.zip': return await _zip(content,fn,ml,meta)
    if ext in ('.tar','.gz','.tgz','.bz2','.xz'): return await _tar(content,fn,ml,meta)
    return FileResult(f"[Archive: {fn} - unsupported]",meta,original_size=len(content))

async def _zip(content,fn,ml,meta):
    files,parts,total=[],[],0
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith('/') or '__MACOSX' in name or name.startswith(('.','__MACOSX')): continue
                try:
                    info=zf.getinfo(name)
                    if info.file_size>MAX_FILE_SIZE or total+info.file_size>MAX_FILE_SIZE*4: continue
                    data=zf.read(name); total+=len(data)
                    if not is_binary_file(name,data):
                        text,_=_decode(data,ml)
                        if text.strip():
                            parts.append(f"\n{'='*60}\nFile: {name}\n{'='*60}\n{text}")
                            files.append({"name":name,"size":len(data),"status":"extracted"})
                except: pass
            txt=f"ZIP: {fn}\nExtracted: {len(parts)}\n\n"+"".join(parts)
            meta.update({"archive_type":"zip","extracted_count":len(parts)})
            if len(txt)>ml: txt=txt[:ml]+"\n\n[... truncated ...]"
            return FileResult(txt,files,meta,len(txt)>ml,len(content))
    except Exception as e:
        return FileResult(f"[ZIP error: {e}]",{**meta,"error":str(e)},original_size=len(content))

async def _tar(content,fn,ml,meta):
    import tarfile; files,parts=[],[]
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            for m in [x for x in tf.getmembers() if x.isfile()]:
                if m.name.startswith(('__MACOSX','.')): continue
                try:
                    f=tf.extractfile(m)
                    if not f: continue
                    data=f.read()
                    if not is_binary_file(m.name,data):
                        text,_=_decode(data,ml)
                        if text.strip():
                            parts.append(f"\n{'='*60}\nFile: {m.name}\n{'='*60}\n{text}")
                            files.append({"name":m.name,"size":m.size,"status":"extracted"})
                except: pass
            txt=f"TAR: {fn}\nExtracted: {len(parts)}\n\n"+"".join(parts)
            if len(txt)>ml: txt=txt[:ml]+"\n\n[... truncated ...]"
            return FileResult(txt,files,{**meta,"extracted_count":len(parts)},len(txt)>ml,len(content))
    except Exception as e:
        return FileResult(f"[TAR error: {e}]",{**meta,"error":str(e)},original_size=len(content))

# =========================
# AUTH & COOKIES
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
BACKUP_COOKIE = "HeloxAI_ID"

def get_user_id(req) -> str:
    uid = (
        req.cookies.get(PRIMARY_COOKIE) or
        req.cookies.get(BACKUP_COOKIE) or
        req.headers.get("X-User-ID") or
        ""
    )
    return uid if uid else str(uuid.uuid4())

def set_session_cookie(response: Response, user_id: str):
    """Set session cookie on response — frontend needs this for conversation persistence"""
    response.set_cookie(
        PRIMARY_COOKIE,
        user_id,
        max_age=SESSION_DURATION,
        httponly=True,
        samesite="lax",
        secure=ENVIRONMENT == "production",
        path="/"
    )

# =========================
# SYSTEM PROMPT
# =========================
BASE_SYS = 'You are HeloXAi1, an AI assistant by GoldYLocks. Use markdown formatting. If asked who made you, say: "I was constructed by GoldYLocks. Find them on Twitter @HeloXAi1"'
CREATOR_INST = ' IMPORTANT: The user asks about your creator. Respond EXACTLY: "I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi1" — nothing else.'
_CPATS = [re.compile(p, re.I) for p in [
    r'who.*(made|created|built|developed|constructed|owns|runs).*you',
    r'your\s+(creator|developer|maker|builder|founder|owner)',
    r'who\s+is\s+behind\s+helox', r'who\s+made\s+helox',
    r'made\s+by\s+who', r'what\s+(company|team)\s+made\s+you',
    r'how\s+were\s+you\s+(made|created|built)'
]]

def sys_prompt(text: str) -> str:
    prompt = BASE_SYS
    if any(p.search(text) for p in _CPATS):
        prompt += CREATOR_INST
    return prompt

# =========================
# TOKEN HELPERS
# =========================
def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1

def calc_max_output(messages: List[Dict]) -> int:
    input_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    safe = max(128, TOKENS_PER_MINUTE_LIMIT - input_tokens - 500)
    return min(safe, MAX_TOKENS)

# =========================
# DB HELPERS (graceful fallback)
# =========================
async def _db_retry(op, desc="DB", retries=2):
    if not supabase: return None
    last = None
    for i in range(retries):
        try:
            return op.execute()
        except Exception as e:
            last = e
        if i < retries - 1:
            await asyncio.sleep(0.1 * (i + 1))
    logger.warning(f"{desc} failed: {last}")
    return None

async def save_conversation(conversation_id: str, user_id: str, title: str):
    if not supabase: return
    try:
        await _db_retry(
            supabase.table("conversations").upsert({
                "id": conversation_id, "user_id": user_id,
                "title": title, "updated_at": datetime.now(timezone.utc).isoformat()
            }, on_conflict="id"), "SaveConv")
    except Exception as e:
        logger.warning(f"save_conv: {e}")

async def save_message(conversation_id: str, user_id: str, role: str, content: str):
    if not supabase: return
    try:
        await _db_retry(
            supabase.table("messages").insert({
                "conversation_id": conversation_id, "user_id": user_id,
                "role": role, "content": content,
                "created_at": datetime.now(timezone.utc).isoformat()
            }), "SaveMsg")
    except Exception as e:
        logger.warning(f"save_msg: {e}")

async def ensure_user_exists(user_id: str):
    if not supabase: return
    try:
        result = await _db_retry(
            supabase.table("users").select("id").eq("id", user_id).limit(1), "CheckUser")
        if not result or not result.data or len(result.data) == 0:
            await _db_retry(
                supabase.table("users").insert({
                    "id": user_id, "anonymous": True, "is_free": True, "plan": "free",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }), "CreateUser")
            logger.info(f"New user: {user_id[:8]}")
    except Exception as e:
        logger.warning(f"ensure_user: {e}")

# =========================
# GROQ NON-STREAMING (reliable on Render)
# =========================
async def call_groq(messages, temperature=None, max_tokens=None) -> str:
    """Call Groq API non-streaming — gets complete response before returning."""
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured")

    temperature = temperature if temperature is not None else TEMPERATURE
    dynamic_max = calc_max_output(messages)
    max_tokens = min(max_tokens or MAX_TOKENS, dynamic_max)

    input_est = estimate_tokens(''.join(m.get('content', '') for m in messages))
    logger.info(f"Groq call: ~{input_est} input tokens, max_out={max_tokens}")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(GROQ_BASE_URL, headers=headers, json=payload)

            if r.status_code == 200:
                data = r.json()
                content = ""
                if data.get("choices") and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                logger.info(f"Groq 200 — {len(content)} chars")
                return content or "[No response]"

            if r.status_code in (413, 429):
                wait = 5 * (attempt + 1)
                logger.warning(f"Groq {r.status_code}, retry {attempt+1}/3 in {wait}s")
                payload["max_tokens"] = max(64, payload["max_tokens"] // 2)
                if attempt < 2:
                    await asyncio.sleep(wait)
                    continue
                return "I'm experiencing high demand. Please try again shortly."

            logger.error(f"Groq {r.status_code}: {r.text[:300]}")
            raise HTTPException(r.status_code, f"Groq error: {r.text[:200]}")

    return "[Error: retries exceeded]"


# =========================
# SSE BUILDER — matches OpenAI format exactly
# Frontend parses: data: {...}\n → JSON.parse → choices[0].delta.content
# =========================
def build_sse(text: str) -> str:
    """
    Build SSE payload in exact OpenAI chunk format.
    Each chunk: data: {json}\n\n
    Final:     data: [DONE]\n\n

    Frontend (from your HTML) expects this exact structure.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    parts = []

    # 1) Role chunk — delta has "role" but NO "content" key
    parts.append(json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }))

    # 2) Content chunks — delta has "content" but NO "role" key
    #    Split by spaces for word-by-word animation in frontend
    words = text.split(' ')
    buf = ""
    for i, w in enumerate(words):
        buf = (buf + " " + w) if buf else w
        if len(buf) >= 5 or i == len(words) - 1:
            parts.append(json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {"content": buf},
                    "finish_reason": None
                }]
            }))
            buf = ""

    # 3) Finish chunk — empty delta, finish_reason="stop"
    parts.append(json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }))

    # Assemble with proper SSE framing
    sse = ""
    for p in parts:
        sse += f"data: {p}\n\n"
    sse += "data: [DONE]\n\n"
    return sse


# =========================
# TAVILY SEARCH
# =========================
async def web_search(query):
    if not TAVILY_API_KEY: return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post("https://api.tavily.com/search",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"},
                json={"query": query, "max_results": 2, "include_answer": True})
            if r.status_code == 200:
                return r.json().get("results", [])
    except Exception as e:
        logger.error(f"Search error: {e}")
    return []


# =========================
# ENDPOINTS
# =========================

@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok", "service": "HeloXAi", "provider": "Groq",
        "model": MODEL_NAME, "model_display": MODEL_DISPLAY, "version": "4.4.0"
    })

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", "provider": "Groq", "model": MODEL_NAME,
        "model_display": MODEL_DISPLAY, "max_tokens": MAX_TOKENS,
        "version": "4.4.0", "groq": bool(GROQ_API_KEY),
        "tavily": bool(TAVILY_API_KEY), "supabase": bool(supabase),
        "cors_origins": FRONTEND_ORIGINS
    }

@app.get("/api/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "groq"}]}


@app.post("/newchat")
async def newchat(request: Request, response: Response):
    """Frontend calls this to create a new chat — returns chat_id for subsequent messages."""
    try:
        body = {}
        try:
            raw = await request.body()
            if raw:
                body = json.loads(raw)
            if not isinstance(body, dict):
                body = {}
        except Exception:
            pass

        conversation_id = str(uuid.uuid4())
        title = body.get("title", "New Chat")
        mode = body.get("mode", "chat")
        user_id = get_user_id(request)

        # Set cookie so frontend can persist sessions
        if not request.cookies.get(PRIMARY_COOKIE) and not request.cookies.get(BACKUP_COOKIE):
            set_session_cookie(response, user_id)

        asyncio.create_task(ensure_user_exists(user_id))
        asyncio.create_task(save_conversation(conversation_id, user_id, title))

        return JSONResponse({
            "chat_id": conversation_id,
            "conversation_id": conversation_id,
            "title": title,
            "mode": mode,
            "model": MODEL_NAME
        })
    except Exception as e:
        logger.error(f"newchat error: {e}", exc_info=True)
        return JSONResponse({
            "chat_id": str(uuid.uuid4()),
            "title": "New Chat", "mode": "chat", "model": MODEL_NAME
        })


@app.post("/ask/universal")
async def ask_universal(request: Request, response: Response):
    """
    MAIN CHAT ENDPOINT — your frontend posts here.

    Request body (any of these keys work):
      - message / prompt / text / content / query / input / msg
      - conversation_id / chat_id / chatId
      - history / messages / context  (array of {role, content})
      - files / attachments  (array of {name, content})
      - use_search / search / web_search  (boolean)
      - temperature, max_tokens

    Response: text/event-stream with OpenAI-compatible SSE chunks.
    """
    try:
        raw_body = await request.body()
        try:
            data = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})
        if not isinstance(data, dict):
            data = {}

        # --- Extract user message (tries all common key names) ---
        user_message = (
            data.get("prompt") or data.get("message") or data.get("msg") or
            data.get("text") or data.get("content") or data.get("input") or
            data.get("query") or ""
        ).strip()

        # Deep search if top-level keys didn't have it
        if not user_message:
            def _find(d, depth=0):
                if depth > 4 or not d: return None
                if isinstance(d, str) and 1 <= len(d.strip()) <= 50000: return d.strip()
                if isinstance(d, dict):
                    for k in ['prompt','message','msg','text','content','input','query']:
                        if k in d:
                            r = _find(d[k], depth + 1)
                            if r: return r
                    for v in d.values():
                        r = _find(v, depth + 1)
                        if r: return r
                if isinstance(d, list):
                    for item in reversed(d):
                        if isinstance(item, dict) and item.get('role') == 'user':
                            c = item.get('content', '').strip()
                            if c: return c
                        r = _find(item, depth + 1)
                        if r: return r
                return None
            user_message = _find(data) or ""

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "No message found", "keys": list(data.keys())})

        logger.info(f"Msg: {user_message[:100]}")

        # --- History (limit to last 4 to save tokens) ---
        history = []
        for key in ['history', 'messages', 'conversation', 'context']:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        role = str(item.get('role', '')).lower()
                        content = item.get('content') or item.get('text') or ''
                        if role in ['user', 'assistant', 'system'] and isinstance(content, str) and content.strip():
                            if len(content) > 500:
                                content = content[:500] + "..."
                            history.append({"role": role, "content": content.strip()})
                break

        # --- Params ---
        use_search = bool(data.get("use_search") or data.get("search") or data.get("web_search"))
        try: temperature = float(data.get("temperature", TEMPERATURE))
        except (ValueError, TypeError): temperature = TEMPERATURE
        try: max_tokens = int(data.get("max_tokens", MAX_TOKENS))
        except (ValueError, TypeError): max_tokens = MAX_TOKENS

        conversation_id = data.get("conversation_id") or data.get("chat_id") or data.get("chatId")
        user_id = get_user_id(request)

        # Set cookie
        if not request.cookies.get(PRIMARY_COOKIE) and not request.cookies.get(BACKUP_COOKIE):
            set_session_cookie(response, user_id)

        # --- File context ---
        file_context = ""
        files = data.get("files") or data.get("attachments") or []
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict):
                    fc = f.get("content") or f.get("text") or ""
                    fn = f.get("name") or f.get("filename") or "file"
                    if fc:
                        if len(fc) > 3000: fc = fc[:3000] + "\n... [truncated]"
                        file_context += f"\n\n--- File: {fn} ---\n{fc}\n--- End ---\n"
        if not file_context and isinstance(data.get("file_content"), str):
            fc = data['file_content']
            if len(fc) > 3000: fc = fc[:3000] + "\n... [truncated]"
            file_context = f"\n\n--- File ---\n{fc}\n--- End ---\n"

        full_message = user_message
        if file_context:
            full_message = f"{user_message}\n\n[Attached Files]{file_context}"
        if len(full_message) > 8000:
            full_message = full_message[:8000] + "\n... [truncated]"

        # --- System prompt ---
        system = sys_prompt(full_message)

        if use_search:
            results = await web_search(full_message)
            if results:
                ctx = "\n\n**Web Search Results:**\n"
                for i, r in enumerate(results[:2], 1):
                    ctx += f"{i}. [{r.get('title','')}]({r.get('url','')})\n   {r.get('content','')[:150]}...\n\n"
                system += ctx

        # --- Build message array for Groq ---
        messages = [{"role": "system", "content": system}]
        for msg in history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": full_message})

        # ====== CRITICAL SECTION ======
        # 1. Call Groq NON-streaming (reliable, no partial reads)
        # 2. Build complete SSE blob
        # 3. Return as plain Response with Content-Length
        #    This prevents Render proxy from buffering chunks indefinitely
        # ================================

        response_text = await call_groq(messages, temperature, max_tokens)

        # Save to DB in background (don't block response)
        if conversation_id:
            asyncio.create_task(save_message(conversation_id, user_id, "user", full_message))
            asyncio.create_task(save_message(conversation_id, user_id, "assistant", response_text))
            asyncio.create_task(save_conversation(conversation_id, user_id, full_message[:80]))

        # Build SSE payload
        sse_payload = build_sse(response_text)
        sse_bytes = sse_payload.encode('utf-8')

        logger.info(f"SSE response: {len(sse_bytes)} bytes")

        return Response(
            content=sse_bytes,
            media_type="text/event-stream; charset=utf-8",
            status_code=200,
            headers={
                # Prevent ALL caching
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                # Tell nginx/Render not to buffer
                "X-Accel-Buffering": "no",
                # Content-Length prevents chunked transfer encoding
                # which Render's proxy can buffer indefinitely
                "Content-Length": str(len(sse_bytes)),
                # Prevent gzip compression which can also buffer
                "Content-Encoding": "identity",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ask/universal error: {type(e).__name__}: {e}", exc_info=True)
        err_sse = build_sse(f"[Error: {str(e)}]")
        err_bytes = err_sse.encode('utf-8')
        return Response(
            content=err_bytes,
            media_type="text/event-stream; charset=utf-8",
            status_code=200,  # Return 200 so frontend still parses SSE
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Content-Length": str(len(err_bytes)),
                "Content-Encoding": "identity",
            }
        )


@app.post("/v1/chat/completions")
async def openai_compatible(request: Request, response: Response):
    """OpenAI-compatible endpoint — works with any OpenAI SDK client."""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages:
            return JSONResponse(status_code=400, content={"error": {"message": "No messages", "type": "invalid_request_error"}})

        stream = data.get("stream", False)
        temperature = data.get("temperature", TEMPERATURE)
        max_tokens = data.get("max_tokens", MAX_TOKENS)

        user_id = get_user_id(request)
        if not request.cookies.get(PRIMARY_COOKIE):
            set_session_cookie(response, user_id)

        # Inject our system prompt if missing
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            messages.insert(0, {"role": "system", "content": sys_prompt(last_user)})

        response_text = await call_groq(messages, temperature, max_tokens)

        if stream:
            sse = build_sse(response_text)
            sse_bytes = sse.encode('utf-8')
            return Response(
                content=sse_bytes,
                media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                         "Content-Length": str(len(sse_bytes)), "Content-Encoding": "identity"}
            )
        else:
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": estimate_tokens(str(messages)), "completion_tokens": estimate_tokens(response_text), "total_tokens": estimate_tokens(str(messages) + response_text)}
            })
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
    except Exception as e:
        logger.error(f"/v1/chat/completions error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})


@app.post("/chat")
async def simple_chat(request: Request):
    """Simple JSON endpoint — returns plain JSON, no SSE."""
    try:
        data = await request.json()
        message = data.get("message") or data.get("prompt") or data.get("text", "")
        if not message:
            return JSONResponse(status_code=400, content={"error": "No message"})
        messages = [{"role": "system", "content": sys_prompt(message)}, {"role": "user", "content": message}]
        response_text = await call_groq(messages)
        return JSONResponse({"response": response_text, "message": response_text})
    except Exception as e:
        logger.error(f"/chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Too large. Max {fmt_size(MAX_FILE_SIZE)}")
        result = await extract_file(content, file.filename)
        return JSONResponse({
            "name": file.filename, "size": len(content),
            "content": result.content, "metadata": result.metadata,
            "truncated": result.truncated
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/stop/{chat_id}")
async def stop_gen(chat_id: str):
    if chat_id in active_streams:
        active_streams[chat_id].cancel()
        del active_streams[chat_id]
    return JSONResponse({"stopped": True})


# =========================
# CORS PREFLIGHT — explicit for all paths
# =========================
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle CORS preflight for every path — prevents 405 on OPTIONS."""
    origin = request.headers.get("Origin", "")
    allowed = origin if origin in FRONTEND_ORIGINS else FRONTEND_ORIGINS[0] if FRONTEND_ORIGINS else "*"

    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": allowed,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
            "Content-Length": "0",
        }
    )


# =========================
# TEST ENDPOINT — verify SSE works without frontend
# =========================
@app.get("/test/sse")
async def test_sse():
    """Hit this in browser to verify SSE format is correct."""
    test_text = "Hello! This is a test response from HeloXAi. **Markdown works!**\n\n- Item 1\n- Item 2\n\n```python\nprint('hello')\n```"
    sse = build_sse(test_text)
    return Response(
        content=sse.encode('utf-8'),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Length": str(len(sse.encode('utf-8'))),
            "Content-Encoding": "identity",
        }
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
