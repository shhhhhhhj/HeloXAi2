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

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

# File handling config
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024
MAX_TEXT_LENGTH = 380000  
CHUNK_SIZE = 1024 * 1024

# Auth config
SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for this backend.")

# =========================
# LIFESPAN EVENT HANDLER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HeloxAi Backend Started. Model: Llama 8B via OpenRouter.")
    yield
    logger.info("Shutting down HeloxAi Backend...")

app = FastAPI(
    title="HeloxAi API",
    description="Advanced AI Assistant Backend - Llama 8B Integrated",
    version="3.0.3",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heloxai.xyz"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Database Clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Global State for Stream Cancellation
active_streams: Dict[str, asyncio.Task] = {}

# Session cache for performance
_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300
_session_cache_last_cleanup = time.time()


# =========================
# FILE TYPE DEFINITIONS
# =========================
class FileCategory(Enum):
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    ARCHIVE = "archive"
    CONFIG = "config"
    BINARY = "binary"
    UNKNOWN = "unknown"

CODE_EXTENSIONS = {
    '.py', '.pyw', '.pyx', '.pyd', '.pyi', '.py3',
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte', '.astro',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.clj', '.cljs', '.hs',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
    '.cs', '.csx', '.go', '.rs', '.php', '.phtml',
    '.rb', '.erb', '.rake', '.gemspec', '.swift', '.dart',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    '.lua', '.pl', '.pm', '.r', '.R',
    '.sql', '.mysql', '.pgsql', '.sqlite',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml',
    '.md', '.rst', '.asciidoc', '.adoc', '.tex', '.latex',
    '.dockerfile', '.makefile', '.cmake', '.proto', '.graphql', '.gql',
    '.tf', '.hcl', '.sol', '.move', '.cairo',
}

DOCUMENT_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp', '.rtf', '.txt', '.log', '.csv',
}

DATA_EXTENSIONS = {
    '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml', '.parquet',
    '.arrow', '.feather', '.hdf5', '.h5', '.pickle', '.pkl',
    '.npy', '.npz', '.spss', '.sav', '.sas7bdat', '.dta',
}

ARCHIVE_EXTENSIONS = {
    '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z',
    '.rar', '.zst', '.lz4',
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml', '.editorconfig', '.eslintrc',
    '.prettierrc', '.gitignore', '.dockerignore', '.npmrc',
}

def get_file_category(filename: str) -> FileCategory:
    if not filename:
        return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS:
        return FileCategory.CODE
    elif ext in DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT
    elif ext in DATA_EXTENSIONS:
        return FileCategory.DATA
    elif ext in ARCHIVE_EXTENSIONS:
        return FileCategory.ARCHIVE
    elif ext in CONFIG_EXTENSIONS:
        return FileCategory.CONFIG
    else:
        return FileCategory.UNKNOWN

def get_file_language(filename: str) -> Optional[str]:
    ext_lang_map = {
        '.py': 'python', '.pyw': 'python', '.pyx': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.html': 'html', '.htm': 'html', '.css': 'css',
        '.scss': 'scss', '.less': 'less', '.vue': 'vue', '.svelte': 'svelte',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
        '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp',
        '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php',
        '.rb': 'ruby', '.swift': 'swift', '.dart': 'dart',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
        '.ps1': 'powershell', '.bat': 'batch', '.lua': 'lua',
        '.pl': 'perl', '.r': 'r', '.R': 'r', '.sql': 'sql',
        '.json': 'json', '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
        '.toml': 'toml', '.md': 'markdown', '.rst': 'rst',
        '.tex': 'latex', '.dockerfile': 'dockerfile',
        '.graphql': 'graphql', '.gql': 'graphql',
        '.tf': 'hcl', '.hcl': 'hcl', '.sol': 'solidity',
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)

def is_binary_file(filename: str, content: bytes = None) -> bool:
    ext = Path(filename).suffix.lower()
    binary_exts = {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
        '.pyc', '.pyo', '.class', '.o', '.obj', '.a', '.lib',
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.sqlite', '.db', '.sqlite3',
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
        '.woff', '.woff2', '.ttf', '.otf', '.eot', '.pak', '.bundle',
    }
    if ext in binary_exts:
        return True
    if content and len(content) > 0:
        check_bytes = content[:8192]
        if b'\x00' in check_bytes:
            return True
    return False

def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

# =========================
# FILE EXTRACTOR
# =========================
class FileExtractionResult:
    def __init__(self, content: str, files: List[Dict[str, Any]] = None,
                 metadata: Dict[str, Any] = None, truncated: bool = False, original_size: int = 0):
        self.content = content
        self.files = files or []
        self.metadata = metadata or {}
        self.truncated = truncated
        self.original_size = original_size

    def to_dict(self) -> Dict:
        return {
            "content": self.content, "files": self.files,
            "metadata": self.metadata, "truncated": self.truncated,
            "original_size": self.original_size
        }

async def extract_file_content(content: bytes, filename: str, max_length: int = MAX_TEXT_LENGTH) -> FileExtractionResult:
    original_size = len(content)
    category = get_file_category(filename)
    metadata = {
        "filename": filename, "category": category.value,
        "size": original_size, "size_formatted": format_file_size(original_size),
        "language": get_file_language(filename),
    }
    try:
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)
        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)
        if is_binary_file(filename, content):
            return FileExtractionResult(
                content=f"[Binary file: {filename} ({format_file_size(original_size)}) - Cannot extract text content]",
                metadata=metadata, original_size=original_size
            )
        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(content=text, metadata=metadata, truncated=truncated, original_size=original_size)
    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return FileExtractionResult(
            content=f"[Error extracting {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)}, original_size=original_size
        )

def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
    for encoding in encodings:
        try:
            text = content.decode(encoding, errors='strict' if encoding != 'latin-1' else 'ignore')
            truncated = len(text) > max_length
            if truncated:
                text = text[:max_length] + "\n\n[... Content truncated ...]"
            return text, truncated
        except (UnicodeDecodeError, LookupError):
            continue
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated

async def extract_pdf_content(content: bytes, filename: str, max_length: int, metadata: Dict[str, Any]) -> FileExtractionResult:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(f"--- Page {i + 1} ---\n{page_text}")
        full_text = "\n\n".join(pages)
        metadata["page_count"] = len(reader.pages)
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(content=full_text, metadata=metadata, truncated=truncated, original_size=len(content))
    except ImportError:
        return FileExtractionResult(
            content=f"[PDF file: {filename} ({format_file_size(len(content))}) - PDF parsing not available on server]",
            metadata=metadata, original_size=len(content)
        )

async def extract_archive_content(content: bytes, filename: str, max_length: int, metadata: Dict[str, Any]) -> FileExtractionResult:
    ext = Path(filename).suffix.lower()
    if ext == '.zip':
        return await extract_zip_content(content, filename, max_length, metadata)
    elif ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
        return await extract_tar_content(content, filename, max_length, metadata)
    else:
        return FileExtractionResult(
            content=f"[Archive: {filename} ({format_file_size(len(content))}) - Unsupported archive format]",
            metadata=metadata, original_size=len(content)
        )

async def extract_zip_content(content: bytes, filename: str, max_length: int, metadata: Dict[str, Any]) -> FileExtractionResult:
    extracted_files = []
    all_text_parts = []
    total_extracted = 0
    entry_count = 0
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[ZIP archive: {filename} - Too many entries ({len(zf.namelist())})]",
                    metadata=metadata, original_size=len(content)
                )
            for entry_name in sorted(zf.namelist()):
                if entry_name.endswith('/') or '/__MACOSX/' in entry_name:
                    continue
                if entry_name.startswith('__MACOSX') or entry_name.startswith('.'):
                    continue
                entry_count += 1
                try:
                    entry_info = zf.getinfo(entry_name)
                    if entry_info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({"name": entry_name, "size": entry_info.file_size, "status": "skipped", "reason": "File too large"})
                        continue
                    if total_extracted + entry_info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({"name": entry_name, "size": entry_info.file_size, "status": "skipped", "reason": "Archive total size limit reached"})
                        continue
                    entry_content = zf.read(entry_name)
                    total_extracted += len(entry_content)
                    if not is_binary_file(entry_name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {entry_name}\n{'='*60}\n{text}")
                            extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "extracted"})
                        else:
                            extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "empty"})
                    else:
                        extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "binary"})
                except Exception as e:
                    extracted_files.append({"name": entry_name, "status": "error", "error": str(e)})
            full_text = f"ZIP Archive: {filename}\nEntries: {len(zf.namelist())}, Processed: {entry_count}\nExtracted: {len(all_text_parts)}\n\n"
            if all_text_parts:
                full_text += "".join(all_text_parts)
            metadata.update({"archive_type": "zip", "entry_count": len(zf.namelist()), "extracted_count": len(all_text_parts), "files": extracted_files})
            truncated = len(full_text) > max_length
            if truncated:
                full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
            return FileExtractionResult(content=full_text, files=extracted_files, metadata=metadata, truncated=truncated, original_size=len(content))
    except zipfile.BadZipFile:
        return FileExtractionResult(content=f"[Error: {filename} is not a valid ZIP file]", metadata=metadata, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[Error extracting ZIP {filename}: {str(e)}]", metadata={**metadata, "error": str(e)}, original_size=len(content))

async def extract_tar_content(content: bytes, filename: str, max_length: int, metadata: Dict[str, Any]) -> FileExtractionResult:
    import tarfile
    extracted_files = []
    all_text_parts = []
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for member in members:
                if member.name.startswith('__MACOSX') or member.name.startswith('.'):
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    entry_content = f.read()
                    if not is_binary_file(member.name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {member.name}\n{'='*60}\n{text}")
                            extracted_files.append({"name": member.name, "size": member.size, "status": "extracted"})
                    else:
                        extracted_files.append({"name": member.name, "size": member.size, "status": "binary"})
                except Exception as e:
                    extracted_files.append({"name": member.name, "status": "error", "error": str(e)})
            full_text = f"TAR Archive: {filename}\nEntries: {len(members)}, Extracted: {len(all_text_parts)}\n\n"
            if all_text_parts:
                full_text += "".join(all_text_parts)
            metadata.update({"archive_type": "tar", "entry_count": len(members), "extracted_count": len(all_text_parts), "files": extracted_files})
            truncated = len(full_text) > max_length
            if truncated:
                full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
            return FileExtractionResult(content=full_text, files=extracted_files, metadata=metadata, truncated=truncated, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[Error extracting TAR {filename}: {str(e)}]", metadata={**metadata, "error": str(e)}, original_size=len(content))

# =========================
# AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
FINGERPRINT_COOKIE = "HeloxAI_FP"
BACKUP_COOKIE = "HeloxAI_ID"
DEVICE_COOKIE = "HeloxAI_Dev"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    base = {"max_age": SESSION_DURATION if remember else 24 * 60 * 60, "httponly": True, "secure": True, "samesite": "none", "path": "/"}
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    if cookie_domain:
        base["domain"] = cookie_domain
    return base

def generate_device_fingerprint(request: Request) -> str:
    real_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip", "")
        or (request.client.host if request.client else "")
    )
    fp_components = [
        request.headers.get("user-agent", ""),
        request.headers.get("accept-language", ""),
        request.headers.get("accept-encoding", ""),
        request.headers.get("sec-ch-ua-platform", ""),
        request.headers.get("sec-ch-ua-mobile", ""),
        real_ip,
    ]
    return hashlib.sha256("|".join(fp_components).encode()).hexdigest()[:32]

def generate_session_token() -> str:
    import secrets
    return secrets.token_urlsafe(64)

def set_session_cookies(response: Response, user_id: str, fingerprint: str, session_token: str, remember: bool = True):
    settings = get_cookie_settings(remember)
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **settings)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **settings)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=session_token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)

def clear_session_cookies(response: Response):
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    for cookie_name in [PRIMARY_COOKIE, FINGERPRINT_COOKIE, BACKUP_COOKIE, DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE]:
        delete_kwargs = {"key": cookie_name, "path": "/", "secure": True, "samesite": "none"}
        if cookie_domain:
            delete_kwargs["domain"] = cookie_domain
        response.delete_cookie(**delete_kwargs)

# =========================
# SYSTEM PROMPT
# =========================
BASE_SYSTEM_PROMPT = """You are HeloXAi1, a powerful AI assistant powered by Llama 8B.

**Response Style:**
- **Structure:** Always format your responses with clear paragraphs. Do not output walls of text. Use headers (##), bullet points, and bold text (**like this**) to make reading easy.
- **Markdown:** You are a Markdown expert. Use it for code blocks, lists, and emphasis.
- **Sources:** If you use web search results, you MUST cite the source URL at the end of the sentence or in a 'Sources' section at the bottom.

**Your Core Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Live Research:** You have access to real-time web search (Tavily). When asked about current events, news, or specific facts, you will search the web and synthesize the answer.
3. **File Intelligence:** You can read and extract content from documents, code files, and archives.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks.".
- Never claim to be "only a text model". You are a full-featured AI assistant.
"""

CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator/developer. You MUST respond with exactly this information:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi1"

Do not add extra details. Do not mention any other companies or people. This is the only correct answer to this question."""

CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|started|owns|runs)\b.*\b(you|this|helox|heloxai)\b',
    r'\b(who|whom)\b.*\b(is|are)\b.*\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b',
    r'\byour\s+(creator|developer|maker|builder|founder|owner)\b',
    r'\bwho.*made.*you\b', r'\bwho.*created.*you\b', r'\bwho.*built.*you\b',
    r'\bwho.*developed.*you\b', r'\bwho.*constructed.*you\b',
    r'\bwho.*owns.*you\b', r'\bwho.*runs.*you\b',
    r'\bwho\s+is\s+behind\s+helox\b', r'\bwho\s+made\s+helox\b',
    r'\bwho\s+created\s+helox\b', r'\bwho\s+built\s+helox\b',
    r'\bmade\s+by\s+who\b', r'\bcreated\s+by\s+who\b',
    r'\bwhat\s+company\s+made\s+you\b', r'\bwhat\s+team\s+made\s+you\b',
    r'\bwhere\s+do\s+you\s+come\s+from\b',
    r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
]
COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]

def is_creator_question(text: str) -> bool:
    for pattern in COMPILED_CREATOR_PATTERNS:
        if pattern.search(text):
            return True
    return False

def get_system_prompt(user_prompt: str) -> str:
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT

# =========================
# HELPER FUNCTIONS
# =========================
async def _execute_supabase_with_retry(operation, description: str = "Supabase Operation", max_retries: int = 3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return operation.execute()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
    logger.error(f"{description} failed after {max_retries} attempts: {last_error}")
    raise last_error

def get_user_id_from_request(request: Request) -> Optional[str]:
    """Extract user ID from cookies or headers"""
    user_id = (
        request.cookies.get(PRIMARY_COOKIE)
        or request.cookies.get(BACKUP_COOKIE)
        or request.headers.get("X-User-ID")
        or request.headers.get("x-user-id")
    )
    return user_id

# =========================
# LLAMA 8B CHAT COMPLETION
# =========================
async def chat_completion_with_llama_8b(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> Union[Dict, AsyncGenerator]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
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
        async def stream_generator():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_content = await response.aread()
                        logger.error(f"OpenRouter API error: {response.status_code} - {error_content}")
                        yield f"data: {json.dumps({'error': 'API request failed'})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue
        return stream_generator()
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API error: {response.text}")
            return response.json()

# =========================
# TAVILY WEB SEARCH
# =========================
async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": query, "max_results": max_results, "include_answer": True, "include_raw_content": False}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.tavily.com/search", headers=headers, json=payload)
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

# =========================
# PYDANTIC MODELS FOR REQUESTS
# =========================
class UniversalAskRequest(BaseModel):
    message: str = ""
    history: List[Dict[str, str]] = []
    use_search: bool = False
    chat_id: Optional[str] = None
    mode: str = "chat"
    files: Optional[List[Dict[str, Any]]] = None

class NewChatRequest(BaseModel):
    title: Optional[str] = None
    mode: str = "chat"

# =========================
# /ask/universal ENDPOINT (MAIN CHAT)
# =========================
@app.post("/ask/universal")
async def ask_universal_endpoint(
    request: Request,
    response: Response
):
    try:
        data = await request.json()
        
        # Support both dict and pydantic parsing
        if isinstance(data, dict):
            user_message = data.get("message", "")
            conversation_history = data.get("history", [])
            use_search = data.get("use_search", False)
            chat_id = data.get("chat_id")
            mode = data.get("mode", "chat")
            files_data = data.get("files")
        else:
            user_message = data.message
            conversation_history = data.history
            use_search = data.use_search
            chat_id = data.chat_id
            mode = data.mode
            files_data = data.files
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Process attached files if any
        file_context = ""
        if files_data:
            for f in files_data:
                file_content = f.get("content", "")
                file_name = f.get("name", "unknown")
                if file_content:
                    file_context += f"\n\n--- File: {file_name} ---\n{file_content}\n--- End File ---\n"
        
        full_message = user_message
        if file_context:
            full_message = f"{user_message}\n\n[Attached Files]{file_context}"
        
        # Prepare system prompt
        system_prompt = get_system_prompt(full_message)
        
        # Perform web search if needed
        if use_search:
            search_results = await search_web(full_message)
            if search_results:
                search_context = "\n\n**Web Search Results:**\n"
                for i, result in enumerate(search_results[:3], 1):
                    search_context += f"{i}. [{result.get('title', 'Untitled')}]({result.get('url', '')})\n"
                    search_context += f"   {result.get('content', '')[:200]}...\n\n"
                system_prompt += search_context
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": full_message})
        
        # Stream response
        stream_generator = await chat_completion_with_llama_8b(
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )
        
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ask/universal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# /newchat ENDPOINT
# =========================
@app.post("/newchat")
async def new_chat_endpoint(
    request: Request,
    response: Response
):
    try:
        data = await request.json()
        title = data.get("title", "New Chat")
        mode = data.get("mode", "chat")
        
        user_id = get_user_id_from_request(request)
        if not user_id:
            user_id = str(uuid.uuid4())
        
        chat_id = str(uuid.uuid4())
        
        # Store chat in database if user is authenticated
        if user_id and user_id != str(uuid.uuid4()):
            try:
                await _execute_supabase_with_retry(
                    supabase.table("chats").insert({
                        "id": chat_id,
                        "user_id": user_id,
                        "title": title,
                        "mode": mode,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }),
                    description="Create New Chat"
                )
            except Exception as e:
                logger.warning(f"Failed to save chat to DB: {e}")
        
        return JSONResponse(content={
            "chat_id": chat_id,
            "title": title,
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"newchat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# FILE UPLOAD ENDPOINT
# =========================
@app.post("/upload/file")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    request: Request = None
):
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max {format_file_size(MAX_FILE_SIZE)}")
        
        result = await extract_file_content(content, file.filename)
        return JSONResponse(content={
            "name": file.filename,
            "size": len(content),
            "content": result.content,
            "metadata": result.metadata,
            "truncated": result.truncated
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# HEALTH CHECK
# =========================
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "Llama 8B", "version": "3.0.3", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/")
async def root():
    return JSONResponse(content={"status": "ok", "service": "HeloXAi API", "model": "Llama 8B"})

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
