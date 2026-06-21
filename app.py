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

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

# Log API key status at startup (without revealing the keys)
logger.info(f"OPENROUTER_API_KEY set: {bool(OPENROUTER_API_KEY)}")
logger.info(f"TAVILY_API_KEY set: {bool(TAVILY_API_KEY)}")

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HeloxAi Backend Started. Model: Llama 8B via OpenRouter.")
    yield
    logger.info("Shutting down HeloxAi Backend...")

app = FastAPI(title="HeloXAi API", description="HeloXAi - Llama 8B", version="3.0.5", lifespan=lifespan)

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
        self.content=content; self.files=files or []; self.metadata=metadata or {}; self.truncated=truncated; self.original_size=original_size
    def to_dict(self): return {"content":self.content,"files":self.files,"metadata":self.metadata,"truncated":self.truncated,"original_size":self.original_size}

async def extract_file(content: bytes, fn: str, ml=MAX_TEXT_LENGTH) -> FileResult:
    sz=len(content); cat=get_file_category(fn); meta={"filename":fn,"category":cat.value,"size":sz,"size_formatted":fmt_size(sz),"language":get_file_language(fn)}
    try:
        if cat==FileCategory.ARCHIVE: return await _extract_arch(content,fn,ml,meta)
        if fn.lower().endswith('.pdf'): return await _extract_pdf(content,fn,ml,meta)
        if is_binary_file(fn,content): return FileResult(f"[Binary: {fn}]",meta,original_size=sz)
        text,trunc=_decode(content,ml); meta["line_count"]=text.count('\n')+1
        return FileResult(text,meta=meta,truncated=trunc,original_size=sz)
    except Exception as e: return FileResult(f"[Error: {e}]",{**meta,"error":str(e)},original_size=sz)

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
    except ImportError: return FileResult(f"[PDF: {fn} - no parser]",meta,original_size=len(content))

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
                    if info.file_size>MAX_FILE_SIZE or total+info.file_size>MAX_EXTRACTED_SIZE: continue
                    data=zf.read(name); total+=len(data)
                    if not is_binary_file(name,data):
                        text,_=_decode(data,ml)
                        if text.strip(): parts.append(f"\n{'='*60}\nFile: {name}\n{'='*60}\n{text}"); files.append({"name":name,"size":len(data),"status":"extracted"})
                except: pass
            txt=f"ZIP: {fn}\nExtracted: {len(parts)}\n\n"+"".join(parts)
            meta.update({"archive_type":"zip","extracted_count":len(parts)})
            if len(txt)>ml: txt=txt[:ml]+"\n\n[... truncated ...]"
            return FileResult(txt,files,meta,len(txt)>ml,len(content))
    except Exception as e: return FileResult(f"[ZIP error: {e}]",{**meta,"error":str(e)},original_size=len(content))

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
                        if text.strip(): parts.append(f"\n{'='*60}\nFile: {m.name}\n{'='*60}\n{text}"); files.append({"name":m.name,"size":m.size,"status":"extracted"})
                except: pass
            txt=f"TAR: {fn}\nExtracted: {len(parts)}\n\n"+"".join(parts)
            if len(txt)>ml: txt=txt[:ml]+"\n\n[... truncated ...]"
            return FileResult(txt,files,{**meta,"extracted_count":len(parts)},len(txt)>ml,len(content))
    except Exception as e: return FileResult(f"[TAR error: {e}]",{**meta,"error":str(e)},original_size=len(content))

# =========================
# AUTH
# =========================
PRIMARY_COOKIE="HeloxAI_Session"; BACKUP_COOKIE="HeloxAI_ID"; SESSION_TOKEN_COOKIE="HeloxAI_Token"
SESSION_EXPIRY_COOKIE="HeloxAI_Expiry"; FINGERPRINT_COOKIE="HeloxAI_FP"; DEVICE_COOKIE="HeloxAI_Dev"

def get_user_id(req): return req.cookies.get(PRIMARY_COOKIE) or req.cookies.get(BACKUP_COOKIE) or req.headers.get("X-User-ID") or req.headers.get("x-user-id")

# =========================
# SYSTEM PROMPT
# =========================
BASE_SYS = """You are HeloXAi1, a powerful AI assistant powered by Llama 8B.

**Response Style:**
- **Structure:** Use headers (##), bullet points, and bold text (**like this**) to make reading easy.
- **Markdown:** Use it for code blocks, lists, and emphasis.
- **Sources:** If you use web search results, cite the source URL.

**Your Core Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Live Research:** Real-time web search via Tavily for current events and facts.
3. **File Intelligence:** Read and extract content from documents, code, and archives.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks."
- Never claim to be "only a text model". You are a full-featured AI assistant.
"""
CREATOR_INST = '\n\nIMPORTANT: The user asks about your creator. Respond EXACTLY: "I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi1" — nothing else.'
_CPATS = [re.compile(p,re.I) for p in [r'who.*(made|created|built|developed|constructed|owns|runs).*you',r'your\s+(creator|developer|maker|builder|founder|owner)',r'who\s+is\s+behind\s+helox',r'who\s+made\s+helox',r'made\s+by\s+who',r'what\s+(company|team)\s+made\s+you',r'how\s+were\s+you\s+(made|created|built)']]

def sys_prompt(text):
    if any(p.search(text) for p in _CPATS): return BASE_SYS+CREATOR_INST
    return BASE_SYS

# =========================
# HELPERS
# =========================
async def _db_retry(op, desc="DB", retries=3):
    last=None
    for i in range(retries):
        try: return op.execute()
        except Exception as e: last=e; 
        if i<retries-1: await asyncio.sleep(0.1*(i+1))
    logger.error(f"{desc} failed: {last}"); raise last

# =========================
# LLAMA 8B — with detailed error logging
# =========================
async def call_llama8b(messages, temperature=0.7, max_tokens=2048, stream=False):
    """Call OpenRouter Llama 8B with full error logging"""
    
    # Pre-flight checks
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set! Cannot call Llama 8B.")
        raise HTTPException(500, "OPENROUTER_API_KEY environment variable is not configured")
    
    logger.info(f"Calling OpenRouter with {len(messages)} messages, temp={temperature}, max_tokens={max_tokens}, stream={stream}")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://heloxai.xyz",
        "X-Title": "HeloXAi"
    }
    
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    if stream:
        async def gen():
            try:
                async with httpx.AsyncClient(timeout=300.0) as c:
                    async with c.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers, json=payload) as r:
                        logger.info(f"OpenRouter stream response status: {r.status_code}")
                        
                        if r.status_code != 200:
                            err_body = await r.aread()
                            logger.error(f"OpenRouter API error {r.status_code}: {err_body.decode('utf-8', errors='replace')[:500]}")
                            yield f"data: {json.dumps({'error': f'API error {r.status_code}'})}\n\n"
                            return
                        
                        async for line in r.aiter_lines():
                            if line.startswith("data: "):
                                d = line[6:]
                                if d.strip() == "[DONE]":
                                    logger.info("OpenRouter stream completed successfully")
                                    yield "data: [DONE]\n\n"
                                    break
                                try:
                                    chunk = json.loads(d)
                                    txt = chunk.get("choices",[{}])[0].get("delta",{}).get("content","")
                                    if txt: yield f"data: {json.dumps({'content': txt})}\n\n"
                                except json.JSONDecodeError:
                                    pass
            except httpx.TimeoutException:
                logger.error("OpenRouter request timed out after 300s")
                yield f"data: {json.dumps({'error': 'Request timed out'})}\n\n"
            except Exception as e:
                logger.error(f"OpenRouter stream error: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return gen()
    else:
        try:
            async with httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                logger.info(f"OpenRouter response status: {r.status_code}")
                if r.status_code != 200:
                    logger.error(f"OpenRouter error body: {r.text[:500]}")
                    raise HTTPException(r.status_code, f"OpenRouter error: {r.text[:200]}")
                return r.json()
        except httpx.TimeoutException:
            logger.error("OpenRouter request timed out")
            raise HTTPException(504, "OpenRouter request timed out")
        except httpx.ConnectError as e:
            logger.error(f"OpenRouter connection error: {e}")
            raise HTTPException(502, f"Cannot connect to OpenRouter: {e}")

# =========================
# TAVILY SEARCH
# =========================
async def web_search(query):
    if not TAVILY_API_KEY: return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post("https://api.tavily.com/search",
                headers={"Content-Type":"application/json","Authorization":f"Bearer {TAVILY_API_KEY}"},
                json={"query":query,"max_results":3,"include_answer":True})
            if r.status_code==200: return r.json().get("results",[])
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
    return {"status":"healthy","model":"Llama 8B","version":"3.0.5","openrouter_key_set":bool(OPENROUTER_API_KEY),"tavily_key_set":bool(TAVILY_API_KEY)}

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
            await _db_retry(supabase.table("chats").insert({
                "id":chat_id,"user_id":user_id,"title":title,"mode":mode,
                "created_at":datetime.now(timezone.utc).isoformat(),
                "updated_at":datetime.now(timezone.utc).isoformat()
            }),"Create Chat")
        except: pass
        return JSONResponse({"chat_id":chat_id,"title":title,"mode":mode})
    except Exception as e:
        logger.error(f"newchat error: {e}", exc_info=True)
        return JSONResponse({"chat_id":str(uuid.uuid4()),"title":"New Chat","mode":"chat"})

@app.post("/ask/universal")
async def ask_universal(request: Request):
    """Main chat endpoint - handles the frontend's exact format:
       {"prompt": "...", "conversation_id": "...", "mode": "general", "model": "helox"}
    """
    try:
        raw_body = await request.body()
        
        # Parse JSON
        try:
            data = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}, body: {raw_body[:200]}")
            return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})
        
        if not isinstance(data, dict):
            data = {}
        
        # === EXTRACT MESSAGE — frontend uses "prompt" field ===
        user_message = (
            data.get("prompt") or          # Frontend's actual field name
            data.get("message") or
            data.get("msg") or
            data.get("text") or
            data.get("content") or
            data.get("input") or
            data.get("query") or
            ""
        ).strip()
        
        if not user_message:
            # Deep search as fallback
            def _find(d, depth=0):
                if depth > 4 or not d: return None
                if isinstance(d, str) and 1 <= len(d.strip()) <= 50000: return d.strip()
                if isinstance(d, dict):
                    for k in ['prompt','message','msg','text','content','input','query']:
                        if k in d:
                            r = _find(d[k], depth+1)
                            if r: return r
                    for v in d.values():
                        r = _find(v, depth+1)
                        if r: return r
                if isinstance(d, list):
                    for item in reversed(d):
                        if isinstance(item, dict) and item.get('role') == 'user':
                            c = item.get('content','').strip()
                            if c: return c
                        r = _find(item, depth+1)
                        if r: return r
                return None
            user_message = _find(data) or ""
        
        if not user_message:
            logger.warning(f"No message found. Keys: {list(data.keys())}")
            return JSONResponse(status_code=400, content={"error":"No message found","keys":list(data.keys())})
        
        logger.info(f"Message: {user_message[:120]}")
        
        # === EXTRACT HISTORY ===
        history = []
        for key in ['history','messages','conversation','context']:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        role = str(item.get('role','')).lower()
                        content = item.get('content') or item.get('text') or ''
                        if role in ['user','assistant','system'] and isinstance(content, str) and content.strip():
                            history.append({"role": role, "content": content.strip()})
                break
        
        # === EXTRACT OPTIONS ===
        use_search = bool(data.get("use_search") or data.get("search") or data.get("web_search"))
        try: temperature = float(data.get("temperature", 0.7))
        except: temperature = 0.7
        try: max_tokens = int(data.get("max_tokens", 2048))
        except: max_tokens = 2048
        conversation_id = data.get("conversation_id") or data.get("chat_id") or data.get("chatId")
        
        # === FILE CONTEXT ===
        file_context = ""
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
        
        # === BUILD SYSTEM PROMPT ===
        system = sys_prompt(full_message)
        
        if use_search:
            results = await web_search(full_message)
            if results:
                ctx = "\n\n**Web Search Results:**\n"
                for i, r in enumerate(results[:3], 1):
                    ctx += f"{i}. [{r.get('title','')}]({r.get('url','')})\n   {r.get('content','')[:200]}...\n\n"
                system += ctx
        
        # === BUILD MESSAGES ===
        messages = [{"role": "system", "content": system}]
        for msg in history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": full_message})
        
        logger.info(f"Sending {len(messages)} messages to Llama 8B...")
        
        # === CALL LLAMA 8B ===
        stream_gen = await call_llama8b(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        
        return StreamingResponse(
            stream_gen, media_type="text/event-stream",
            headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ask/universal UNHANDLED: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {str(e)}"})

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE: raise HTTPException(413, f"Too large. Max {fmt_size(MAX_FILE_SIZE)}")
        result = await extract_file(content, file.filename)
        return JSONResponse({"name":file.filename,"size":len(content),"content":result.content,"metadata":result.metadata})
    except HTTPException: raise
    except Exception as e: logger.error(f"Upload error: {e}", exc_info=True); raise HTTPException(500, str(e))

@app.post("/stop/{chat_id}")
async def stop_gen(chat_id: str):
    if chat_id in active_streams: active_streams[chat_id].cancel(); del active_streams[chat_id]
    return JSONResponse({"stopped": True})

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
