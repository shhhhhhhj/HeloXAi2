import os
import re
import json
import base64
import uuid
import asyncio
import logging
import hashlib
import zipfile
import mimetypes
import time
import tempfile

import httpx
from supabase import create_client, create_async_client
from fastapi import UploadFile, File, Form 
import numpy as np
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple, AsyncGenerator
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Cookie, Header, Form
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel


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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024
MAX_TEXT_LENGTH = 380000

SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HeloxAi Backend Started. Model: Llama 8B via OpenRouter.")
    yield
    logger.info("Shutting down HeloxAi Backend...")

app = FastAPI(
    title="HeloXAi API",
    description="HeloXAi - Llama 8B",
    version="3.0.4",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
active_streams: Dict[str, asyncio.Task] = {}

# =========================
# FILE TYPES
# =========================
class FileCategory(Enum):
    CODE = "code"; DOCUMENT = "document"; DATA = "data"
    ARCHIVE = "archive"; CONFIG = "config"; BINARY = "binary"; UNKNOWN = "unknown"

CODE_EXTENSIONS = {
    '.py','.pyw','.pyx','.js','.jsx','.mjs','.ts','.tsx','.html','.css','.scss',
    '.vue','.svelte','.java','.kt','.scala','.c','.cpp','.cs','.go','.rs',
    '.php','.rb','.swift','.dart','.sh','.bash','.sql','.json','.yaml','.yml',
    '.toml','.md','.dockerfile','.graphql','.tf','.hcl','.sol',
}
DOCUMENT_EXTENSIONS = {'.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.txt','.csv','.rtf'}
DATA_EXTENSIONS = {'.csv','.tsv','.json','.xml','.yaml','.parquet','.pkl','.npy'}
ARCHIVE_EXTENSIONS = {'.zip','.tar','.gz','.tgz','.bz2','.xz','.7z','.rar'}
CONFIG_EXTENSIONS = {'.json','.yaml','.yml','.toml','.ini','.cfg','.conf','.env','.xml'}

def get_file_category(filename: str) -> FileCategory:
    if not filename: return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS: return FileCategory.CODE
    if ext in DOCUMENT_EXTENSIONS: return FileCategory.DOCUMENT
    if ext in DATA_EXTENSIONS: return FileCategory.DATA
    if ext in ARCHIVE_EXTENSIONS: return FileCategory.ARCHIVE
    if ext in CONFIG_EXTENSIONS: return FileCategory.CONFIG
    return FileCategory.UNKNOWN

def get_file_language(filename: str) -> Optional[str]:
    m = {'.py':'python','.js':'javascript','.ts':'typescript','.html':'html','.css':'css',
         '.vue':'vue','.java':'java','.cpp':'cpp','.go':'go','.rs':'rust','.sql':'sql',
         '.json':'json','.yaml':'yaml','.md':'markdown','.sh':'bash','.swift':'swift'}
    return m.get(Path(filename).suffix.lower())

def is_binary_file(filename: str, content: bytes = None) -> bool:
    ext = Path(filename).suffix.lower()
    if ext in {'.exe','.dll','.so','.bin','.zip','.tar','.gz','.7z','.rar','.pdf',
                '.doc','.docx','.xls','.xlsx','.png','.jpg','.jpeg','.gif','.webp',
                '.mp3','.mp4','.wav','.avi','.mov','.mkv','.woff','.ttf','.sqlite'}: return True
    if content and len(content) > 0 and b'\x00' in content[:8192]: return True
    return False

def format_file_size(s: int) -> str:
    for u in ['B','KB','MB','GB']:
        if s < 1024.0: return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{s:.1f} TB"

# =========================
# FILE EXTRACTOR
# =========================
class FileExtractionResult:
    def __init__(self, content: str, files=None, metadata=None, truncated=False, original_size=0):
        self.content = content; self.files = files or []; self.metadata = metadata or {}
        self.truncated = truncated; self.original_size = original_size
    def to_dict(self):
        return {"content": self.content, "files": self.files, "metadata": self.metadata,
                "truncated": self.truncated, "original_size": self.original_size}

async def extract_file_content(content: bytes, filename: str, max_length=MAX_TEXT_LENGTH) -> FileExtractionResult:
    sz = len(content)
    cat = get_file_category(filename)
    meta = {"filename": filename, "category": cat.value, "size": sz, "size_formatted": format_file_size(sz), "language": get_file_language(filename)}
    try:
        if cat == FileCategory.ARCHIVE: return await _extract_archive(content, filename, max_length, meta)
        if filename.lower().endswith('.pdf'): return await _extract_pdf(content, filename, max_length, meta)
        if is_binary_file(filename, content):
            return FileExtractionResult(f"[Binary: {filename}]", meta, original_size=sz)
        text, trunc = _decode_text(content, max_length)
        meta["line_count"] = text.count('\n') + 1
        return FileExtractionResult(text, meta=meta, truncated=trunc, original_size=sz)
    except Exception as e:
        return FileExtractionResult(f"[Error: {e}]", {**meta, "error": str(e)}, original_size=sz)

def _decode_text(content: bytes, max_len: int) -> Tuple[str, bool]:
    for enc in ['utf-8','utf-8-sig','latin-1','cp1252']:
        try:
            t = content.decode(enc, errors='strict' if enc != 'latin-1' else 'ignore')
            if len(t) > max_len: t = t[:max_len] + "\n\n[... truncated ...]"
            return t, len(t) > max_len
        except: continue
    t = content.decode('utf-8', errors='replace')
    if len(t) > max_len: t = t[:max_len] + "\n\n[... truncated ...]"
    return t, len(t) > max_len

async def _extract_pdf(content, fn, ml, meta):
    try:
        from PyPDF2 import PdfReader
        pages = [f"--- Page {i+1} ---\n{p.extract_text() or ''}" for i, p in enumerate(PdfReader(BytesIO(content)).pages)]
        txt = "\n\n".join(pages)
        meta["page_count"] = len(pages)
        if len(txt) > ml: txt = txt[:ml] + "\n\n[... truncated ...]"
        return FileExtractionResult(txt, meta=meta, truncated=len(txt)>ml, original_size=len(content))
    except ImportError:
        return FileExtractionResult(f"[PDF: {fn} - parser not available]", meta, original_size=len(content))

async def _extract_archive(content, fn, ml, meta):
    ext = Path(fn).suffix.lower()
    if ext == '.zip': return await _extract_zip(content, fn, ml, meta)
    if ext in ('.tar','.gz','.tgz','.bz2','.xz'): return await _extract_tar(content, fn, ml, meta)
    return FileExtractionResult(f"[Archive: {fn} - unsupported]", meta, original_size=len(content))

async def _extract_zip(content, fn, ml, meta):
    files, parts, total = [], [], 0
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith('/') or '__MACOSX' in name or name.startswith(('.', '__MACOSX')): continue
                try:
                    info = zf.getinfo(name)
                    if info.file_size > MAX_FILE_SIZE or total + info.file_size > MAX_EXTRACTED_SIZE: continue
                    data = zf.read(name); total += len(data)
                    if not is_binary_file(name, data):
                        text, _ = _decode_text(data, ml)
                        if text.strip():
                            parts.append(f"\n{'='*60}\nFile: {name}\n{'='*60}\n{text}")
                            files.append({"name": name, "size": len(data), "status": "extracted"})
                except: pass
            txt = f"ZIP: {fn}\nEntries: {len(zf.namelist())}, Extracted: {len(parts)}\n\n" + "".join(parts)
            meta.update({"archive_type": "zip", "extracted_count": len(parts)})
            if len(txt) > ml: txt = txt[:ml] + "\n\n[... truncated ...]"
            return FileExtractionResult(txt, files, meta, len(txt)>ml, len(content))
    except Exception as e:
        return FileExtractionResult(f"[ZIP error: {e}]", {**meta,"error":str(e)}, original_size=len(content))

async def _extract_tar(content, fn, ml, meta):
    import tarfile
    files, parts = [], []
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            for m in [x for x in tf.getmembers() if x.isfile()]:
                if m.name.startswith(('__MACOSX','.')): continue
                try:
                    f = tf.extractfile(m)
                    if not f: continue
                    data = f.read()
                    if not is_binary_file(m.name, data):
                        text, _ = _decode_text(data, ml)
                        if text.strip():
                            parts.append(f"\n{'='*60}\nFile: {m.name}\n{'='*60}\n{text}")
                            files.append({"name": m.name, "size": m.size, "status": "extracted"})
                except: pass
            txt = f"TAR: {fn}\nExtracted: {len(parts)}\n\n" + "".join(parts)
            if len(txt) > ml: txt = txt[:ml] + "\n\n[... truncated ...]"
            return FileExtractionResult(txt, files, {**meta, "extracted_count": len(parts)}, len(txt)>ml, len(content))
    except Exception as e:
        return FileExtractionResult(f"[TAR error: {e}]", {**meta,"error":str(e)}, original_size=len(content))

# =========================
# AUTH
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
BACKUP_COOKIE = "HeloxAI_ID"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"
FINGERPRINT_COOKIE = "HeloxAI_FP"
DEVICE_COOKIE = "HeloxAI_Dev"

def get_user_id(request: Request) -> Optional[str]:
    return (request.cookies.get(PRIMARY_COOKIE) or request.cookies.get(BACKUP_COOKIE)
            or request.headers.get("X-User-ID") or request.headers.get("x-user-id"))

# =========================
# SYSTEM PROMPT
# =========================
BASE_SYSTEM_PROMPT = """You are HeloXAi1, a powerful AI assistant powered by Llama 8B.

**Response Style:**
- **Structure:** Always format your responses with clear paragraphs. Use headers (##), bullet points, and bold text (**like this**).
- **Markdown:** You are a Markdown expert. Use it for code blocks, lists, and emphasis.
- **Sources:** If you use web search results, cite the source URL.

**Your Core Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Live Research:** Real-time web search via Tavily for current events and facts.
3. **File Intelligence:** Read and extract content from documents, code, and archives.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks."
- Never claim to be "only a text model". You are a full-featured AI assistant.
"""

CREATOR_INSTRUCTION = '\n\nIMPORTANT: The user asks about your creator. Respond EXACTLY: "I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi1" — nothing else.'
_CREATOR_PATTERNS = [re.compile(p, re.I) for p in [
    r'who.*(made|created|built|developed|constructed|owns|runs).*you',
    r'your\s+(creator|developer|maker|builder|founder|owner)',
    r'who\s+is\s+behind\s+helox', r'who\s+made\s+helox', r'made\s+by\s+who',
    r'what\s+(company|team)\s+made\s+you', r'how\s+were\s+you\s+(made|created|built)',
]]

def get_system_prompt(text: str) -> str:
    if any(p.search(text) for p in _CREATOR_PATTERNS):
        return BASE_SYSTEM_PROMPT + CREATOR_INSTRUCTION
    return BASE_SYSTEM_PROMPT

# =========================
# HELPERS
# =========================
async def _supabase_retry(op, desc="DB", retries=3):
    last = None
    for i in range(retries):
        try: return op.execute()
        except Exception as e:
            last = e
            if i < retries - 1: await asyncio.sleep(0.1 * (i + 1))
    logger.error(f"{desc} failed: {last}")
    raise last

def _deep_find_message(data: Any, depth: int = 0) -> Optional[str]:
    """Recursively search for a string that looks like a user message"""
    if depth > 5: return None
    if isinstance(data, str):
        s = data.strip()
        if len(s) >= 1 and len(s) < 50000: return s
        return None
    if isinstance(data, dict):
        # Priority keys to check first
        for key in ['message','msg','text','content','prompt','input','query','question','user_message','userMessage','body','data','value']:
            if key in data:
                result = _deep_find_message(data[key], depth + 1)
                if result and len(result) >= 1: return result
        # Then check all other keys
        for k, v in data.items():
            if k in ['history','messages','conversation','context','files','attachments','mode','chat_id','chatId','temperature','max_tokens','use_search','search','web_search']:
                continue
            result = _deep_find_message(v, depth + 1)
            if result and len(result) >= 1: return result
    if isinstance(data, list):
        # Look for the last string in the list that could be a message
        for item in reversed(data):
            if isinstance(item, str) and len(item.strip()) >= 1:
                return item.strip()
            if isinstance(item, dict):
                # Check if it's a chat message object
                role = item.get('role', '')
                content = item.get('content') or item.get('text') or item.get('message', '')
                if role == 'user' and isinstance(content, str) and content.strip():
                    return content.strip()
    return None

def _deep_find_history(data: Any, depth: int = 0) -> List[Dict]:
    """Extract conversation history from the payload"""
    if depth > 3: return []
    if isinstance(data, list):
        history = []
        for item in data:
            if isinstance(item, dict):
                role = str(item.get('role', '')).lower()
                content = item.get('content') or item.get('text') or item.get('message') or ''
                if role in ['user','assistant','system'] and isinstance(content, str) and content.strip():
                    history.append({"role": role, "content": content.strip()})
        return history
    if isinstance(data, dict):
        for key in ['history','messages','conversation','context','chat_history']:
            if key in data:
                result = _deep_find_history(data[key], depth + 1)
                if result: return result
    return []

# =========================
# LLAMA 8B
# =========================
async def llama8b_chat(messages: List[Dict], temperature=0.7, max_tokens=2048, stream=False):
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://heloxai.xyz",
        "X-Title": "HeloXAi"
    }
    payload = {"model": "meta-llama/llama-3-8b-instruct", "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens, "stream": stream}
    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=300.0) as c:
                async with c.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                                    headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        err = await r.aread()
                        logger.error(f"OpenRouter {r.status_code}: {err}")
                        yield f"data: {json.dumps({'error': 'API error'})}\n\n"
                        return
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            d = line[6:]
                            if d.strip() == "[DONE]": yield "data: [DONE]\n\n"; break
                            try:
                                chunk = json.loads(d)
                                txt = chunk.get("choices",[{}])[0].get("delta",{}).get("content","")
                                if txt: yield f"data: {json.dumps({'content': txt})}\n\n"
                            except: continue
        return gen()
    else:
        async with httpx.AsyncClient(timeout=300.0) as c:
            r = await c.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if r.status_code != 200: raise HTTPException(r.status_code, f"OpenRouter: {r.text}")
            return r.json()

# =========================
# TAVILY SEARCH
# =========================
async def web_search(query: str) -> List[Dict]:
    if not TAVILY_API_KEY: return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post("https://api.tavily.com/search",
                headers={"Content-Type":"application/json","Authorization":f"Bearer {TAVILY_API_KEY}"},
                json={"query":query,"max_results":3,"include_answer":True})
            if r.status_code == 200: return r.json().get("results",[])
    except Exception as e: logger.error(f"Search error: {e}")
    return []

# =========================
# ENDPOINTS
# =========================

@app.get("/")
async def root():
    return JSONResponse({"status":"ok","service":"HeloXAi","model":"Llama 8B"})

@app.get("/api/health")
async def health():
    return {"status":"healthy","model":"Llama 8B","version":"3.0.4"}

@app.post("/newchat")
async def newchat(request: Request):
    try:
        body = {}
        try:
            raw = await request.body()
            if raw: body = json.loads(raw)
            if not isinstance(body, dict): body = {}
        except: pass
        chat_id = str(uuid.uuid4())
        title = body.get("title", "New Chat")
        mode = body.get("mode", "chat")
        user_id = get_user_id(request) or str(uuid.uuid4())
        try:
            await _supabase_retry(
                supabase.table("chats").insert({
                    "id": chat_id, "user_id": user_id, "title": title, "mode": mode,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }), "Create Chat")
        except: pass
        return JSONResponse({"chat_id": chat_id, "title": title, "mode": mode})
    except Exception as e:
        logger.error(f"newchat error: {e}")
        return JSONResponse({"chat_id": str(uuid.uuid4()), "title": "New Chat", "mode": "chat"})

@app.post("/ask/universal")
async def ask_universal(request: Request):
    """Ultra-permissive chat endpoint that accepts ANY format"""
    try:
        content_type = request.headers.get("content-type", "")
        raw_body = await request.body()
        
        # LOG EVERYTHING for debugging
        logger.info(f"=== /ask/universal RECEIVED ===")
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Body length: {len(raw_body)}")
        logger.info(f"Body preview: {raw_body[:500]}")
        
        data = None
        user_message = None
        
        # 1. Try JSON parse
        if raw_body:
            try:
                data = json.loads(raw_body)
                logger.info(f"Parsed JSON type: {type(data).__name__}")
                if isinstance(data, dict):
                    logger.info(f"JSON keys: {list(data.keys())}")
            except json.JSONDecodeError:
                logger.info("Not valid JSON")
                # Try as plain text
                try:
                    text = raw_body.decode('utf-8').strip()
                    if text and len(text) > 0:
                        user_message = text
                        logger.info(f"Using raw body as message: {text[:100]}")
                except: pass
        
        # 2. Try form data
        if user_message is None and content_type and "form" in content_type.lower():
            try:
                form = await request.form()
                logger.info(f"Form fields: {list(form.keys())}")
                for key in ['message','msg','text','content','prompt','input','query']:
                    if key in form:
                        val = form[key]
                        user_message = val if isinstance(val, str) else str(val)
                        break
                if user_message is None:
                    # Just use the first text field
                    for k, v in form.items():
                        if isinstance(v, str) and v.strip():
                            user_message = v
                            break
            except Exception as e:
                logger.info(f"Form parse failed: {e}")
        
        # 3. Deep search in parsed JSON for message
        if user_message is None and isinstance(data, dict):
            user_message = _deep_find_message(data)
            if user_message:
                logger.info(f"Deep found message: {user_message[:100]}")
        
        # 4. If data is a string (not dict), use it directly
        if user_message is None and isinstance(data, str):
            user_message = data.strip()
        
        # 5. If data is a list, look for the last user message
        if user_message is None and isinstance(data, list):
            for item in reversed(data):
                if isinstance(item, str):
                    user_message = item
                    break
                if isinstance(item, dict) and item.get('role') == 'user':
                    user_message = item.get('content','').strip()
                    if user_message: break
        
        # FINAL CHECK
        if not user_message or not user_message.strip():
            logger.error("=== COULD NOT FIND MESSAGE IN REQUEST ===")
            logger.error(f"Raw body was: {raw_body[:1000]}")
            return JSONResponse(
                status_code=400,
                content={"error": "No message found", "received_keys": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                         "hint": "Send JSON like: {\"message\": \"hello\"}"}
            )
        
        user_message = user_message.strip()
        logger.info(f"Final message to send: {user_message[:150]}")
        
        # Extract history
        history = _deep_find_history(data) if isinstance(data, (dict, list)) else []
        
        # Extract options
        use_search = False
        temperature = 0.7
        max_tokens = 2048
        if isinstance(data, dict):
            use_search = bool(data.get("use_search") or data.get("search") or data.get("web_search"))
            try: temperature = float(data.get("temperature", 0.7))
            except: pass
            try: max_tokens = int(data.get("max_tokens", 2048))
            except: pass
        
        # Extract file context
        file_context = ""
        if isinstance(data, dict):
            files = data.get("files") or data.get("attachments") or []
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict):
                        fc = f.get("content") or f.get("text") or ""
                        fn = f.get("name") or f.get("filename") or "file"
                        if fc: file_context += f"\n\n--- File: {fn} ---\n{fc}\n--- End ---\n"
            if not file_context and isinstance(data.get("file_content"), str):
                file_context = f"\n\n--- File ---\n{data['file_content']}\n--- End ---\n"
        
        full_message = user_message
        if file_context:
            full_message = f"{user_message}\n\n[Attached Files]{file_context}"
        
        # Build system prompt with search
        system_prompt = get_system_prompt(full_message)
        if use_search:
            results = await web_search(full_message)
            if results:
                ctx = "\n\n**Web Search Results:**\n"
                for i, r in enumerate(results[:3], 1):
                    ctx += f"{i}. [{r.get('title','')}]({r.get('url','')})\n   {r.get('content','')[:200]}...\n\n"
                system_prompt += ctx
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": full_message})
        
        # Stream
        stream_gen = await llama8b_chat(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        return StreamingResponse(stream_gen, media_type="text/event-stream",
                                  headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})
    
    except HTTPException: raise
    except Exception as e:
        logger.error(f"ask/universal error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Too large. Max {format_file_size(MAX_FILE_SIZE)}")
        result = await extract_file_content(content, file.filename)
        return JSONResponse({"name": file.filename, "size": len(content), "content": result.content, "metadata": result.metadata})
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.post("/stop/{chat_id}")
async def stop_gen(chat_id: str):
    if chat_id in active_streams:
        active_streams[chat_id].cancel()
        del active_streams[chat_id]
    return JSONResponse({"stopped": True})

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
