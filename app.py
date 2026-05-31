import os
import re
import json
import base64
import uuid
import asyncio
import logging
import hashlib
import zipfile
import tempfile
import mimetypes
import shutil
from fastapi import UploadFile, File, Form 
import cv2  
import numpy as np
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Cookie, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from fastapi.responses import PlainTextResponse
import time

import httpx
from supabase import create_client, create_async_client

# NEW: Google Generative AI Import
# CHANGED: Switched to Legacy Google AI (OpenRouter compatible)
# This allows using Qwen models via OpenRouter
import google.generativeai as genai

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
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # CRITICAL: Used for backend Admin access

# CHANGED: Switched to Legacy Google AI (OpenRouter compatible)
# This allows using Qwen models via OpenRouter
import google.generativeai as genai

# Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY").strip() if os.getenv("GOOGLE_API_KEY") else None

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # NEW: For live research & images
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

# File handling config
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB for zips
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024  # 200MB total extracted
MAX_TEXT_LENGTH = 380000  
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for large files

# Auth config
SESSION_DURATION = 365 * 24 * 60 * 60  # 1 year in seconds
REFRESH_THRESHOLD = 7 * 24 * 60 * 60  # Refresh session if less than 7 days remaining

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # Service Key is required for the Backend API to bypass RLS for custom cookie auth
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for this backend.")

app = FastAPI(
    title="HeloxAi API",
    description="Advanced AI Assistant Backend",
    version="2.7.0" # Switched to Legacy Google Generative AI (OpenRouter Compatible)
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
_session_cache_ttl = 300  # 5 minutes
_session_cache_last_cleanup = time.time()

# =========================
# GOOGLE AI CONFIGURATION
# =========================
# FIXED: Defaulting to text-embedding-3.3-32B (OpenRouter compatible)
# This allows using Qwen models via OpenRouter
# Qwen 3.3-32B is a strong embedding and general purpose model (DeepSeeker-V2.5).
# It handles both tool use and chat competently.
GOOGLE_AI_MODEL = os.getenv("GOOGLE_AI_MODEL", "text-embedding-3.3-32B")

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    logger.info(f"Google Generative AI configured successfully via Client (Legacy). Model: {GOOGLE_AI_MODEL}")
else:
    logger.warning("GOOGLE_API_KEY not set. Chat features will fail.")
    client = None

# =========================
# FILE TYPE DEFINITIONS
# =========================
class FileCategory(Enum):
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    CONFIG = "config"
    BINARY = "binary"
    UNKNOWN = "unknown"

# Comprehensive file type mappings
CODE_EXTENSIONS = {
    # Python
    '.py', '.pyw', '.pyx', '.pyx', '.pyd', '.pyi', '.py3',
    # JavaScript/TypeScript
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    # Web
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte', '.astro',
    # Java/JVM
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.clj', '.cljs', '.hs',
    # C/C++
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
    # C#
    '.cs', '.csx',
    # Go
    '.go',
    # Rust
    '.rs',
    # PHP
    '.php', '.phtml',
    # Ruby
    '.rb', '.erb', '.rake', '.rake', '.gemspec',
    # Swift
    '.swift',
    # Dart/Flutter
    '.dart',
    # Shell
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    # Lua
    '.lua',
    # Perl
    '.pl', '.pm',
    # R
    '.r', '.R',
    # SQL
    '.sql', '.mysql', '.pgsql', '.sqlite',
    # Config/Data formats
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml',
    # Markup
    '.md', '.rst', '.asciidoc', '.adoc', '.adoc', '.tex', '.latex',
    # Other
    '.dockerfile', '.makefile', '.cmake', '.proto', '.graphql', '.gql', '.hcl',
    '.tf', '.hcl', '.sol', '.move', '.cairo',
]

DOCUMENT_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp', '.rtf', '.txt', '.log', '.csv',
}

DATA_EXTENSIONS = {
    '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.parquet', '.arrow', '.feather', '.hdf5', '.h5', '.pickle', '.pkl',
    '.npy', '.npz', '.spss', '.sav', '.sas7bdat', '.dta',
}

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp',
    '.ico', '.tiff', '.tif', '.avif', '.heic', '.heif',
}

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    '.opus', '.aiff', '.ape',
}

VIDEO_EXTENSIONS = {
    '.mp4', '.webm', '.avi', '.mov', '.mkv', '.mkv', '.flv', '.wmv',
    '.m4v', '.ogv', '.3gp',
}

ARCHIVE_EXTENSIONS = {
    '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z',
    '.rar', '.zst', '.lz4',
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml', '.editorconfig', '.eslintrc', '.prettierrc', '.prettierrc', '.gitignore', '.dockerignore', '.npmrc', '.npmrc',
}

def get_file_category(filename: str) -> FileCategory:
    """Determine file category from extension"""
    if not filename:
        return FileCategory.UNKNOWN
    
    ext = Path(filename).suffix.lower()
    
    if ext in CODE_EXTENSIONS:
        return FileCategory.CODE
    elif ext in DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT
    elif ext in DATA_EXTENSIONS:
        return FileCategory.DATA
    elif ext in IMAGE_EXTENSIONS:
        return FileCategory.IMAGE
    elif ext in AUDIO_EXTENSIONS:
        return FileCategory.AUDIO
    elif ext in VIDEO_EXTENSIONS:
        return FileCategory.VIDEO
    elif ext in ARCHIVE_EXTENSIONS:
        return FileCategory.ARCHIVE
    elif ext in CONFIG_EXTENSIONS:
        return FileCategory.CONFIG
    else:
        return FileCategory.UNKNOWN

def get_file_language(filename: str) -> Optional[str]:
    """Get programming language from file extension for syntax highlighting"""
    ext_lang_map = {
        '.py': 'python', '.pyw': 'python', '.pyx': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.html': 'html', '.htm', 'html',
        '.css': 'css', '.scss', '.less', 'less', '.styl',
        '.vue': 'vue', '.svelte', '.astro',
        '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
        '.clj', '.cljs', '.hs',
        '.c': 'c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
        '.cs': 'csharp',
        '.go',
        '.rs',
        '.rb', '.erb', '.rake', '.gemspec',
        '.swift',
        '.dart',
        '.ts', '.dart',
        '.sh', '.bash', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.lua',
        '.pl', '.pm',
        '.r',
        '.rs', '.R',
        '.sql', '.mysql', '.pgsql', '.sqlite',
        '.yaml', '.yml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.env', '.properties', '.xml', '.editorconfig', '.eslintrc', '.prettierrc', '.prettierrc', '.gitignore', '.dockerignore', '.npmrc',
        '.prettierrc',
        '.prettierrc',
        '.pyproject',
        '.makefile', '.cmake', '.proto', '.proto',
        '.graphql', '.gql', '.gql',
        '.orm', 'sequelize', '.sqlalchemy', '.typeorm', '.drizzle',
        '.prisma', '.sqlalchemy', '.orm', '.drizzle'
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)

def is_binary_file(filename: str, content: bytes = None) -> bool:
    """Check if file is binary based on extension or content"""
    ext = Path(filename).suffix.lower()
    
    # Known binary extensions
    binary_exts = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
        '.pyc', '.pyo', '.class', '.o', '.obj', '.a', '.lib',
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.pptx',
        '.sqlite', '.db', '.sqlite3',
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.tiff', '.tif', '.avif', '.heic', '.heif',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flv', '.wmv',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.pak', '.bundle',
    }
    
    if ext in binary_exts:
        return True
    
    # Check content for null bytes (indicates binary)
    if content and len(content) > 0:
        # Check first 8192 bytes for null bytes
        check_bytes = content[:8192]
        if b'\x00' in check_bytes:
            return True
    
    return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

# =========================
# ADVANCED FILE EXTRACTOR
# =========================
class FileExtractionResult:
    def __init__(
        self,
        content: str,
        files: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
        truncated: bool = False,
        original_size: int = 0
    ):
        self.content = content
        self.files = files or []
        self.metadata = metadata or {}
        self.truncated = truncated
        self.original_size = original_size

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "files": self.files,
            "metadata": self.metadata,
            "truncated": self.truncated,
            "original_size": self.original_size
        }

async def extract_file_content(
    content: bytes,
    filename: str,
    max_length: int = MAX_TEXT_LENGTH
) -> FileExtractionResult:
    """
    Extract text content from any file type.
    """
    original_size = len(content)
    category = get_file_category(filename)
    metadata = {
        "filename": filename,
        "version": category.value,
        "size": original_size,
        "size_formatted": format_file_size(original_size),
        "language": get_file_language(filename),
    }

    try:
        # Handle archives (zip files)
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)

        # Handle images
        if category == FileCategory.IMAGE:
            return FileExtractionResult(
                content=f"[Image file: {filename} ({format_file_size(original_size)}) - Use image analysis endpoint for visual content]",
                metadata=metadata,
                original_size=len(content)
            )

        # Handle audio/video
        if category in (FileCategory.AUDIO, FileCategory.VIDEO):
            return FileExtractionResult(
                content=f"[{category.value.capitalize()} file: {filename} ({format_file_size(original_size)}) - Media file cannot be extracted as text]",
                metadata=metadata,
                original_size=len(content)
            )

        # Handle PDF
        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)

        # Handle code and text files
        if category in (FileCategory.CODE, FileCategory.CONFIG, FileCategory.UNKNOWN):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=len(content)
            )

        # Handle documents and data files
        if category in (FileCategory.DOCUMENT, FileCategory.DATA):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=len(content)
            )

        # Handle binary files
        if is_binary_file(filename, content):
            return FileExtractionResult(
                content=f"[Binary file: {filename} ({format_file_size(original_size)}) - Cannot extract text content]",
                metadata=metadata,
                original_size=len(content)
            )

        # Fallback: try to decode as text
        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(
            content=text,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )

    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return FileExtractionResult(
            content=f"[Error extracting {filename}: {str(e)}",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )

def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    """Extract text with multiple encoding fallbacks"""
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
    
    # Final fallback: decode with error replacement
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated

async def extract_pdf_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract text from PDF files"""
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
        
        return FileExtractionResult(
            content=full_text,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
    except ImportError:
        logger.warning("PyPDF2 not installed, returning placeholder for PDF")
        return FileExtractionResult(
            content=f"[PDF file: {filename} ({format_file_size(len(content))}) - PDF parsing not available on server]",
            metadata=metadata,
            original_size=len(content)
        )
    
async def extract_archive_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract and read text contents from TAR archives"""
    import tarfile
    
    extracted_files = []
    all_text_parts = []
    
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            
            if len(members) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[TAR archive: {filename} - Too many entries ({len(members)}) - Maximum allowed: {MAX_ZIP_ENTRIES}]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            for member in members:
                if member.name.startswith('./') or member.name.startswith('/'):
                    member.name = member.name.lstrip('./')
                
                if member.name.startswith('__MACOSX') or member.name.startswith('.'):
                    continue
                
                try:
                    f = tf.extractfile(member)
                    if not is_binary_file(member.name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {member.name}\n{'='*60}\n{text}")
                            extracted_files.append({
                                "name": member.name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "language": entry_language,
                                "status": "extracted",
                                "line_count": text.count('\n') + 1,
                                "preview": text[:500] + ("..." if len(text) > 500 else "")
                            })
                            extracted_files.append({
                                "name": member.name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "status": "extracted",
                                "line_count": text.count('\n') + 1,
                                "preview": text[:500] + ("..." if len(text) > 500 else "")
                            else:
                                extracted_files.append({
                                    "name": member.name,
                                    "size": len(entry_content),
                                    "size": len(entry_content),
                                    "size_formatted": format_file_size(len(entry_content)),
                                    "category": entry_category.value,
                                    "status": "empty",
                                    "note": "File is empty"
                                })
                else:
                    extracted_files.append({
                        "name": member.name,
                        "size": member.size,
                        "size": format_file_size(member.size),
                        "status": "binary",
                        "category": entry_category.value,
                        "status": "binary",
                        "note": "Binary file - cannot extract text"
                    })
            
                except Exception as e:
                    extracted_files.append({
                        "name": member.name,
                        "status": "error",
                        "error": str(e)
                    })
        
        full_text = f"TAR Archive: {filename}\n"
        full_text += f"Entries: {len(members)}, Extracted: {len(all_text_parts)}\n\n"
        
        metadata.update({
            "archive_type": "tar",
            "entry_count": len(members),
            "extracted_count": len(all_text_parts),
            "files": extracted_files
        })
        
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        
        return FileExtractionResult(
            content=full_text,
            files=extracted_files,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
        
    except Exception as e:
        return FileExtractionResult(
            content=f"[Error extracting TAR {filename}: {str(e)}",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )

# =========================
# PRODUCTION-GRADE AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
FINGERPRINT_COOKIE = "HeloxAI_FP"
BACKUP_COOKIE = "HeloxAI_ID"
DEVICE_COOKIE = "HeloxAI_Dev"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"

# Cookie settings - production grade
def get_cookie_settings(remember: bool = True) -> Dict:
    """Get cookie settings based on remember preference"""
    base = {
        "max_age": SESSION_DURATION if remember else 24 * 60 * 60,
        "httponly": True,
        "secure": True,
        "samesite": "none",
        "path": "/"
    }
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    if cookie_domain:
        base["domain"] = cookie_domain
    return base

def generate_device_fingerprint(request: Request) -> str:
    """Generate a stable device fingerprint from request headers"""
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
    fp_string = "|".join(fp_components)
    return hashlib.sha256(fp_string.encode()).hexdigest()[:32]

def generate_session_token() -> str:
    """Generate a secure session token"""
    import secrets
    return secrets.token_urlsafe(64)

def set_session_cookies(
    response: Response,
    user_id: str,
    fingerprint: str,
    session_token: str,
    remember: bool = True
):
    """Set all session cookies for maximum persistence"""
    settings = get_cookie_settings(remember)
    
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **settings)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **settings)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=session_token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)

def clear_session_cookies(response: Response):
    """Clear all session cookies on logout"""
    cookies_to_clear = [
        PRIMARY_COOKIE, FINGERPRINT_COOKIE, BACKUP_COOKIE,
        DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE,
    ]
    
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    
    for cookie_name in cookies_to_clear:
        delete_kwargs = {
            "key": cookie_name,
            "path": "/",
            "secure": True,
            "samesite": "none",
        }
        if cookie_domain:
            delete_kwargs["domain"] = cookie_domain
        response.delete_cookie(**delete_kwargs)
        response.delete_cookie(**delete_kwargs)

def is_session_expired(expiry_str: str) -> bool:
    """Check if session has expired"""
    try:
        expiry = int(expiry_str)
        return time.time() > expiry
    except (ValueError, TypeError):
        return True

def should_refresh_session(expiry_str: str) -> bool:
    """Check if session should be refreshed"""
    try:
        expiry = int(expiry_str)
        remaining = expiry - time.time()
        return remaining < REFRESH_THRESHOLD
    except (ValueError, TypeError):
        return True

async def validate_session_token(user_id: str, token: str) -> bool:
    """Validate session token against stored value"""
    try:
        if user_id in _session_cache:
            cached = _session_cache[user_id]
            if cached.get("token") == token:
                return True
        
        result = await _execute_supabase_with_retry(
            supabase.table("user_sessions")
            .select("token, expires_at")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1),
            description="Validate Session Token"
        )
        
        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {
                "token": token,
                "expires_at": result.data[0].get("expires_at")
            }
            return True
        
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

async def create_user_session(
    user_id: str,
    fingerprint: str,
    remember: bool = True
) -> str:
    """Create a new user session in the database"""
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=SESSION_DURATION if remember else 24 * 60 * 60
    )
    
    try:
        await _execute_supabase_with_retry(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "token": token,
                "fingerprint": fingerprint,
                "user_agent": "",
                "ip_address": "",
                "expires_at": expires_at.isoformat(),
                "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create User Session"
        )
        
        _session_cache[user_id] = {
            "token": token,
            "expires_at": expires_at.isoformat()
        }
        
        return token
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return token

async def cleanup_session_cache():
    """Periodically clean up expired session cache entries"""
    global _session_cache_last_cleanup
    global _session_cache_last_cleanup
    now = time.time()
    
    if now - _session_cache_last_cleanup < _session_cache_ttl:
        return
    
    _session_cache_last_cleanup = now
    expired_keys = []
    
    for user_id, data in _session_cache.items():
        expires_at = data.get("expires_at")
        if expires_at:
            try:
                expiry_time = datetime.fromisoformat(expires_at).timestamp()
                if now > expiry_time:
                    expired_keys.append(user_id)
            except:
                expired_keys.append(user_id)
    
    for key in expired_keys:
        del _session_cache[key]

# =========================
# BASE SYSTEM PROMPT (UPDATED)
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi2, a powerful, multi-modal AI assistant.

**Response Style:**
- **Structure:** Always format your responses with clear paragraphs. Do not output walls of text. Use headers (##), bullet points, and bold text (**like this**) to make reading easy.
- **Markdown:** You are a Markdown expert. Use it for code blocks, lists, and emphasis.
- **Sources:** If you use web search results, you MUST cite the source URL at the end of the sentence or in a 'Sources' section at the bottom.

**Your Core Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Image Generation:** You can generate images from descriptions.
3. **Live Research:** You have access to real-time web search (Tavily). When asked about current events, news, or specific facts, you will search the web and synthesize the answer.
4. **File Intelligence:** You can read and extract content from documents.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi".
- Never claim to be "only a text model". You are a full multi-modal assistant.
"""

CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator/developer. You MUST respond with exactly this information:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi".

Do not add extra details. Do not mention any other companies or people. This is the only correct answer to this question."""

CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|started|starts|owns|runs)\b.*\b(you|whom)\b.*\b(is|are|does|do)\s+(this|my|the|it)\s+(what|is|does|mean))\s+(what|is)\s+(what|why)\s+(how)\s+(to))\s+(how\s+(to|can\s+i)\s+(write|create|implement|build|code|build)\s+(a\s+)?(\w+\s+)?(\w+\s+)?(\w+\s+)?(\w+\s+)?)?\s+(\w+\s+)?',
    r'\b(who\s+is\s+are\s+(you|the)\s+(creator|developer|maker|builder|founded|started|starts|owns|runs)\b.*\b(who|whom)\s+(made|created|built|developed|constructed|programmed|designed|founded|started|starts|owns|runs)\b.*\b(who.*made.*you\b.*\b(who.*built.*you\b)\b(who.*created.*you\b)\b(who.*built.*you\b)\b(who.*programmed.*you\b)\b(who.*developed.*you)\b(who.*constructed.*you\b)\b(who.*owns\s+runs)\b.*\b(who.*made.*you\b)\b(who.*programmed.*you\b)\b(who.*runs)\b.\s+(w+\s+)i?\s+(do|can\s+i)\s+(write|create|implement)\s+(a\s+)?(\w+\s+)?.\s+)?.\s+build)\s+(a\s+)?\s+)?\s+provide?\s+(a\s+)?\s+(full\s+ready|runnable)\s+code.\s*)\s+(a\s+ready\s+use?\s+(\w+\s+?,?\s+?\s+how\s+(to|can\s+i)\s+(write|create|implement)\s+(a\s+)?\s+(w+\s+?,\s+model\s+generate_content)\s+(w+\s+?,\s+max_output_tokens)?\s+.",
            "\n\n# ...\nYou are also an expert software engineer. When writing code:\n1. Write clean, well-structured, production-ready code\n2. Include appropriate error handling\n3. Add helpful comments for complex logic\n4. Consider edge cases\n5. Follow language-specific conventions and best practices\nAlways provide complete, runnable code when possible.""",

            IntentCategory.CODE_DEBUG: """

            IntentCategory.CODE_DEBUG: """

            IntentCategory.CODE_REVIEW: """

You are also a senior code reviewer. Provide constructive feedback on:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Best practices and design patterns
5. Security concerns
Be specific and actionable in your suggestions.""",

            IntentCategory.CODE_GENERATION: """

You are also an expert API development expert. When creating APIs:
1. Follow RESTful principles (or GraphQL best practices)
2. Include proper error handling and status codes
3. Add input validation
4. Add input validation
5. Document endpoints clearly
6. Document endpoints clearly
7. OpenAPI/Swagger/OpenAPI documentation
8. System prompt updates""",
        return base + sub_prompts.get(intent.intent, "\n\nYou are also a helpful coding assistant.")

# Singleton instance
_detector = None

def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None:
        _detector = AdvancedIntentDetector()
    return _detector

# =========================
# BACKWARD COMPATIBLE FUNCTIONS
# =========================
def is_image_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "image"

def is_video_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "video"

def is_code_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "code"

def is_document_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "document"

def is_data_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "data"

# =========================
# NEW ADVANCED FUNCTIONS
# =========================
def detect_intent(prompt: str) -> Optional[IntentResult]:
    return get_detector().get_primary_intent(prompt)

def get_action_type(prompt: str) -> str:
    return get_detector().get_action_type(prompt)

def get_required_tools(prompt: str) -> List[str]:
    return get_detector().get_required_tools(prompt)

def is_debug_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_DEBUG

def is_review_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_REVIEW

def get_intent_confidence(prompt: str) -> float:
    intent = detect_intent(prompt)
    return intent.confidence if intent else 0.0

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True  # New: persist session

class RegenerateRequest(BaseModel):
    conversation_id: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"

class IntentInfo(BaseModel):
    intent: str
    confidence: float
    sub_intents: List[str]
    action_type: str
    tools: List[str]

class FileAnalysisResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    files: List[Dict[str, Any]] = []
    truncated: bool = False

# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def _execute_supabase_with_retry(query_builder, description="Supabase Operation"):
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(query_builder.execute)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            if "502" in error_str or "Bad Gateway" in error_str or "Expecting value" in error_str:
                logger.warning(f"{description} encountered transient error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            else:
                logger.error(f"{description} failed: {e}")
                break
    
    if last_exception:
        raise last_exception

async def get_user(
    request: Request,
    response: Response,
    remember: Optional[bool] = None
) -> Dict[str, Any]:
    await cleanup_session_cache()
    
    # Get all cookie values
    primary_id = request.cookies.get(PRIMARY_COOKIE)
    backup_id = request.cookies.get(BACKUP_COOKIE)
    device_cookie = request.cookies.get(DEVICE_COOKIE)
    stored_fingerprint = request.cookies.get(FINGERPRINT_COOKIE)
    session_token = request.cookies.get(SESSION_TOKEN_COOKIE)
    session_expiry = request.cookies.get(SESSION_EXPIRY_COOKIE)
    
    # Generate current fingerprint
    current_fingerprint = generate_device_fingerprint(request)
    
    # Determine remember preference
    if remember is None:
        # Default to True unless session is about to expire
        remember = not is_session_expired(session_expiry or "0")
    
    user_obj = {
        "id": None,
        "email": None,
        "memory": "",
        "fingerprint": current_fingerprint,
        "session_valid": False,
        "session_token": None
    }

    # Priority 1: Validate existing session token
    user_id = None
    if primary_id and session_token:
        # Check if session is expired
        if is_session_expired(session_expiry or "0"):
            logger.info(f"Session expired for user {primary_id[:8]}...")
            clear_session_cookies(response)
        else:
            # Validate token
            token_valid = await validate_session_token(primary_id, session_token)
            if token_valid:
                user_id = primary_id
                user_obj["session_valid"] = True
                user_obj["session_token"] = session_token
                
                # Check if session should be refreshed
                if should_refresh_session(session_expiry or "0"):
                    logger.info(f"Refreshing session for user {user_id[:8]}...")
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
                    user_obj["session_valid"] = True
            else:
                logger.warning(f"Invalid session token for user {primary_id[:8]}...")
    
    # Priority 2: Try backup cookie
    if not user_id and backup_id:
        user_id = backup_id
        logger.info(f"User recovered via backup cookie: {user_id[:8]}...")
    
    # Priority 3: Try device fingerprint lookup
    if not user_id and device_cookie:
        try:
            fp_part = device_cookie.split("_")[0] if "_" in device_cookie else device_cookie
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1),
                description="User Lookup by Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via device fingerprint: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Fingerprint lookup failed: {e}")

    # Priority 4: Try stored fingerprint
    if not user_id and stored_fingerprint:
        try:
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", stored_fingerprint).limit(1),
                description="User Lookup by Stored Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via stored fingerprint: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Stored fingerprint lookup failed: {e}")

    # Priority 5: Recover user by CURRENT fingerprint (no cookie needed)
    if not user_id and current_fingerprint:
        try:
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users")
                .select("id")
                .eq("fingerprint", current_fingerprint)
                .order("created_at", desc=False)   # get the ORIGINAL user, not a dupe
                .limit(1)
                description="User Lookup by Current Fingerprint (cookie-free)"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via current fingerprint (no cookie): {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Current fingerprint lookup failed: {e}")

    # Load user data if we found an ID
    if user_id:
        try:
            user_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", user_id).limit(1),
                description="User Lookup by ID"
            )
            if user_resp.data:
                u = user_resp.data[0]
                user_obj = {
                    "id": u["id"],
                    "email": u.get("email"),
                    "memory": u.get("memory", ""),
                    "is_premium": u.get("is_premium", False),
                    "is_lifetime": u.get("is_lifetime", False),
                    "plan": u.get("plan", "free"),
                    "fingerprint": current_fingerprint,
                    "session_valid": user_obj.get("session_valid", False),
                    "session_token": user_obj.get("session_token")
                }
                
                # Update fingerprint if changed
                if u.get("fingerprint") != current_fingerprint:
                    try:
                        await _execute_supabase_with_retry(
                            supabase.table("users").update({"fingerprint": current_fingerprint}).eq("id", user_id),
                            description="Update Fingerprint"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update fingerprint: {e}")
                
                # Create or refresh session
                if not user_obj["session_valid"]:
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
                    user_obj["session_valid"] = True
                
                # Set all cookies
                set_session_cookies(response, user_id, current_fingerprint, user_obj["session_token"], remember)
                
                logger.info(f"User authenticated: {user_id[:8]}... session_valid={user_obj['session_valid']}")
                return user_obj
        except Exception as e:
            logger.error(f"User data fetch failed: {e}")
            raise HTTPException(500, "Speech to Text failed")

    # Create new anonymous user
    new_id = str(uuid.uuid4())
    
    try:
        new_user_data = {
            "id": new_id,
            "email": f"anon+{new_id[:8]}@local",
            "memory": "",
            "fingerprint": current_fingerprint
        }
        
        await _execute_supabase_with_retry(
            supabase.table("users").upsert(new_user_data, on_conflict="id"),
            description="Create Anonymous User"
        )
        
        user_obj["id"] = new_id

    # Create session for new user
    new_token = await create_user_session(new_id, current_fingerprint, remember)
    user_obj["session_token"] = new_token
    user_obj["session_valid"] = True
    
    # Set all cookies
    set_session_cookies(response, new_id, current_fingerprint, new_token, remember)
    
    logger.info(f"New user created: {new_id[:8]}... with fingerprint {current_fingerprint[:8]}...")
    
    return user_obj

async def update_user_memory(user_id: str, old_memory: str, user_prompt: str, assistant_response: str):
    """
    Uses an LLM to intelligently update the user's long-term memory.
    Updated to use Google Gemini (Legacy) via the new Client.
    """
    if not client:
        return

    # System prompt for the internal Memory Agent
    memory_agent_prompt = """You are a memory management AI. Update the user's long-term memory based on the latest interaction.

Rules:
1. Retain permanent user facts (Name, Job, Preferences).
2. Update the current context/topic (e.g., "User is discussing HTML").
3. Be concise (max 250 words).
4. Discard conversational filler like "The user said..." or "I responded...".
5. Maintain continuity. If the topic shifts, acknowledge both old and new contexts briefly.
6. Return ONLY the new memory string."""

    user_message = f"""Current Memory:
{old_memory if old_memory else "[Empty]"

Latest Interaction:
User: {user_prompt}
Assistant: {assistant_response}

Updated Memory:"""

    try:
        def run_gen():
            return client.models.generate_content(
                model=GOOGLE_AI_MODEL,
                contents=user_message,
                config={
                    "system_instruction": memory_agent_prompt,
                    "generation_config": {"max_output_tokens": 300, "temperature": 0.1}
                }
            )
        
        response = await asyncio.to_thread(run_gen)
        new_memory_content = response.text.strip()

        # Update Database
        await _execute_supabase_with_retry(
            supabase.table("users").update({"memory": new_memory_content}).eq("id", user_id),
            description="Update User Memory (Intelligent)"
        )
        
        # Update Cache
        if user_id in _session_cache:
            _session_cache[user_id]["memory"] = new_memory_content
        
        logger.info(f"Memory successfully updated for user {user_id[:8]}...")
        return

    except Exception as e:
        logger.error(f"Memory update failed unexpectedly for {user_id[:8]}: {e}")
        return

def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# =========================
# WEB SEARCH INTEGRATION (TAVILY) - UPDATED
# =========================
async def perform_web_search(query: str) -> Dict[str, Any]:
    """Performs a web search using Tavily API and returns formatted results + images."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set.")
        return {"text_context": "[Search unavailable]", "images": []}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
                "include_images": True,  # CRITICAL: This fetches Google images
                "include_raw_content": False
            }
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # 1. Format Text Context for the LLM
            formatted_results = []
            if "answer" in data and data["answer"]:
                formatted_results.append(f"Direct Answer: {data['answer']}\n")
            
            for result in data.get("results", []):
                formatted_results.append(
                    f"Source Title: {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Content: {result['content']\n"
                )
            
            text_context = "\n".join(formatted_results)

            # 2. Extract Images for the Frontend
            images = data.get("images", [])
            
            return {
                "text_context": text_context,
                "images": images # List of image URLs
            }
            
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"text_context": "[Error performing search]", "images": []}

# =========================
# VIDEO WATERMARK SYSTEM
# =========================
async def fetch_logo_image() -> Optional[bytes]:
    """Fetch the logo.png from the configured URL"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(LOGO_URL)
            if response.status_code == 200:
                return response.content
            logger.error(f"Failed to fetch logo: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Logo fetch error: {e}")
    return None

# =========================
# VIDEO PROCESSING HELPERS
# =========================

def get_video_duration(video_bytes: bytes) -> float:
    """
    Returns the duration of the video in seconds using OpenCV.
    Writes bytes to a temp file for reliable reading.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        if fps == 0:
            return 0.0 # Prevent division by zero
            
        duration = frame_count / fps
        return duration
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def extract_video_frames(video_bytes: bytes, max_frames: int = 4) -> list:
    """
    Extracts base64 encoded frames from video bytes.
    Returns a list of base64 strings (jpeg format).
    """
    frames_b64 = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(frame_b64)
                
        cap.release()
        return frames_b64
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
        return []
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

async def add_watermark_to_video(video_url: str) -> str:
    """Add transparent watermark to video"""
    try:
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
        import tempfile
        
        logo_bytes = await fetch_logo_image()
        if not logo_bytes:
            logger.warning("No logo available, returning unwatermarked video")
            return video_url
        
        async with httpx.AsyncClient(timeout=120) as client:
            video_response = await client.get(video_url)
            if video_response.status_code != 200:
                logger.error(f"Failed to download video: HTTP {video_response.status_code}")
                return video_url
            video_bytes = video_response.content
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            logo_path = os.path.join(tmpdir, "logo.png")
            output_path = os.path.join(tmpdir, "output.mp4")
            
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            with open(logo_path, "wb") as f:
                f.write(logo_bytes)
            
            def process_video():
                video = VideoFileClip(video_path)
                logo_width = int(video.w * 0.15)
                logo = ImageClip(logo_path)
                logo_aspect = logo.h / logo.w
                logo_height = int(logo_width * logo_aspect)
                logo_height = int(logo_width * logo_aspect)
                logo = logo.resize((logo_width, logo_height))
                padding = 20
                logo = logo.set_position((video.w - logo_width - padding, video.h - logo_height - padding))
                logo = logo.set_duration(video.duration)
                logo = logo.set_opacity(0.7)
                final = CompositeVideoClip([video, logo])
                final.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=os.path.join(tmpdir, "temp_audio.m4a"),
                    remove_temp=True,
                    logger=None
                )
                video.close()
                logo.close()
                final.close()
                return output_path
            
            output_path = await asyncio.to_thread(process_video)
            
            with open(output_path, "rb") as f:
                watermarked_bytes = f.read()
            
            filename = f"watermarked_{uuid.uuid4().hex}.mp4"
            path = f"public/videos/{filename}"
            
            try:
                await asyncio.to_thread(
                    lambda: supabase.storage.from_("ai-videos").upload(
                        path, watermarked_bytes, {"content-type": "video/mp4"}
                    )
                watermarked_url = f"{SUPABASE_URL}/storage/v1/object/public/ai-videos/{path}"
                logger.info(f"Watermarked video uploaded: {watermarked_url}")
                return watermarked_url
            except Exception as upload_err:
                logger.warning(f"Storage upload failed, using data URI: {upload_err}")
                b64_video = base64.b64encode(watermarked_bytes).decode()
                return f"data:video/mp4;base64,{b64_video}"

    except ImportError:
        logger.error("moviepy not installed")
        return video_url
    except Exception as e:
        logger.error(f"Watermark error: {e}")
        return video_url

# =========================
# =========================
# MAIN ENDPOINTS
# =========================
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    return Response(status_code=200)

@app.get("/robots.txt")
def robots():
    return PlainTextResponse("User-agent: *\nDisallow:")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "HeloxAi Backend",
        "version": "2.7.0",
        "features": {
            "intent_detection": "advanced",
            "user_recognition": "production-grade",
            "file_handling": "comprehensive",
            "session_management": "persistent",
            "memory": "intelligent_llm_consolidation",
            "chat_management": "global_sorted",
            "media_generation": "fixed_and_optimized",
            "web_search": "tavily_with_images",
            "llm_backend": "google_gemini",
            "model": "google_genai"
        }
    }

# =========================
# MEDIA GENERATION HANDLERS (FIXED)
# =========================

async def handle_image_generation(prompt: str, user: Dict[str, Any, conv_id: str, stream: bool, style: str = None, size: str = "1024x1024"):
    """
    FIXED: Updated to use DALL-E-3 for better quality and reliability.
    """
    if not OPENAI_API_KEY:
        msg = "OpenAI API Key not configured."
        async def err_gen(): yield sse({"type": "error", "message": msg})
        if stream: return StreamingResponse(err_gen(), media_type="text/event-stream")
        return {"error": msg}
    
    if not prompt or not prompt.strip(): 
        raise HTTPException(400, "Prompt is required")
    
    if len(prompt) > 4000: 
        prompt = prompt[:4000]

    # DALL-E-3 has 'natural' and 'vivid' styles.
    quality = "standard"
    api_style = "vivid" 
    
    if style == "realistic" or style == "natural":
        api_style = "natural"
    elif style:
        prompt = f"{prompt}, {style} style"
    
    try:
        async with httpx.AsyncClient(timeout=90) as client_http:
            r = await client_http.post(
                "https://api.openai.com/v1/images/generations", 
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, 
                json={
                    "model": "dall-e-3", 
                    "prompt": prompt, 
                    "size": size, 
                    "quality": quality,
                    "style": api_style,
                    "n": 1,
                    "response_format": "url"
                }
            )
            
            if r.status_code != 200:
                logger.error(f"OpenAI Image Error: {r.text}")
                
            r.raise_for_status()
            data = r.json()
            
            images = []
            try:
                for item in data.get("data", []):
                    url = item.get("url")
                    revised_prompt = item.get("revised_prompt", prompt)
                    if url:
                        images.append({"url": url, "revised_prompt": revised_prompt})
                        
            except Exception as e:
                logger.error(f"Failed to parse image response: {e}")
                msg = "Failed to process generated image."
                async def err_gen(): yield sse({"type": "error", "message": msg})
                if stream: return StreamingResponse(err_gen(), media_type="text/event-stream")
                return {"error": msg}
        
    except Exception as e:
        logger.error(f"Image gen unexpected error: {e}")
        async def err_gen(): yield sse({"type": "error", "message": str(e)})
        if stream: return StreamingResponse(err_gen(), media_type="text/event-stream")
        return {"error": str(e)}

@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "echo", "name": "Echo"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"}
        ]
    }

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Fixed STT: Complete implementation with optimized httpx usage.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI Key missing")

    content = await file.read()
    
    # Optimized: Use a reasonable timeout and efficient async client
    async with httpx.AsyncClient(timeout=30.0) as client_http:
        files = {"file": (file.filename, content, file.content_type)}
        data = {"model": "whisper-1"}
        
        try:
            r = await client_http.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files, 
                data=data
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"STT Error: {e.response.text}")
            raise HTTPException(e.response.status_code, f"STT Failed: {e.response.text}")
        except Exception as e:
            logger.error(f"STT Exception: {e}")
            raise HTTPException(500, "Speech to Text failed")

# =========================
# UPDATED ANALYSIS ENDPOINT
# =========================
@app.post("/analysis")
async def analyze_files(
    req: Request,
    file: List[UploadFile] = File(...),
    prompt: str = Form(""),
    stream: bool = True
):
    """
    Enhanced file analysis endpoint supporting:
    - Up to 5 images at once.
    - 1 video (max 1 minute duration).
    - User text prompt to guide the analysis.
    """
    user = await get_user(req, Response())
    
    if len(file) > 5:
        raise HTTPException(400, "Maximum of 5 files allowed at a time.")

    visual_items = []
    text_items = []
    metadata_list = []

    video_count = 0

    for uploaded_file in file:
        content = await uploaded_file.read()
        filename = uploaded_file.filename or "unknown"
        content_type = uploaded_file.content_type or ""
        file_size = len(content)
        
        logger.info(f"[FILE] Upload: {filename} ({format_file_size(file_size)}, type={content_type})")

        if not content:
            continue 

        # --- VIDEO HANDLING ---
        if content_type.startswith("video/") or filename.lower().endswith(('.mp4', '.mov', '.webm', '.avi', '.avi'):
            video_count += 1
            if video_count > 1:
                raise HTTPException(400, "Only 1 video can be analyzed at a time.")
            
            try:
                duration = get_video_duration(content)
                if duration > 60:
                    raise HTTPException(400, f"Video is too long ({duration:.1f}s). Maximum allowed is 60 seconds.")
                logger.info(f"[VIDEO] Duration accepted: {duration:.1f}s")
            except Exception as e:
                logger.error(f"Video processing error: {e}")
                raise HTTPException(400, "Could not process video file. Ensure it is a valid format.")

            frames = extract_video_frames(content)
            for i, frame_b64 in enumerate(frames):
                visual_items.append({'type': 'video', 'b64': frame_b64, 'frame_index': i})

        # --- IMAGE HANDLING ---
        elif content_type.startswith("image/") or get_file_category(filename) == FileCategory.IMAGE:
            b64 = base64.b64encode(content).decode()
            visual_items.append({'type': 'image', 'b64': b64})

        # --- TEXT/ARCHIVE/CODE HANDLING ---
        else:
            category = get_file_category(filename)
            max_allowed = MAX_ZIP_SIZE if category == FileCategory.ARCHIVE else MAX_FILE_SIZE
            
            if file_size > max_allowed:
                raise HTTPException(400, f"File {filename} too large.")

            result = await extract_file_content(content, filename)
            
            text_items.append(f"--- FILE: {filename} ---\n{result.content}")
            metadata_list.append(result.metadata)

    if visual_items:
        logger.info(f"[ANALYSIS] Processing {len(visual_items)} visual items. User prompt: '{prompt[:50]}...'")
        return await handle_visual_analysis(visual_items, stream, user_prompt=prompt)

    if text_items:
        combined_text = "\n\n".join(text_items)
        
        if prompt:
            instruction = f"USER INSTRUCTION: {prompt}\n\n"
            combined_text = instruction + combined_text
        else:
            instruction = f"Analyze the following {len(text_items)} file(s)." if len(text_items) > 1 else "Analyze the following file."
            combined_text = instruction + combined_text
        
        return await handle_text_analysis(
            combined_text,
            stream,
            file_metadata={"note": instruction, "files": metadata_list},
            user_prompt=prompt
        )

    raise HTTPException(400, "No valid files provided for analysis.")

# Helper for visual analysis (images/video frames)
async def handle_visual_analysis(visual_items: list, stream: bool, user_prompt: str = ""):
    """
    Constructs a multi-modal prompt for LLM to analyze multiple images/video frames.
    Currently uses GPT-4o-mini as it was specifically optimized for vision in this codebase.
    """
    content_parts = [
        {"type": "text", "text": user_prompt or "Analyze these visual items in detail. Describe everything you see."}
    ]
    
    for item in visual_items:
        item_type = item.get('type', 'image')
        b64_data = item.get('b64', '')
        
        if item_type == 'image':
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
            })
        elif item_type == 'video':
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
            })
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": content_parts
        }]
    }

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                async with httpx.AsyncClient(timeout=60.0) as client_http:
                    r = await client_http.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=get_openai_headers(),
                        json=payload
                    )
                    r.raise_for_status()
                
                result = r.json()["choices"][0]["message"]["content"]
                yield sse({"type": "text", "text": result})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Visual analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})

        return StreamingResponse(gen(), media_type="text/event-stream")
    
    return {"analysis": "Visual analysis completed."}

async def handle_archive_analysis(
    result: FileExtractionResult,
    stream: bool
):
    """Special handling for archive files with multiple extracted files"""
    
    files_summary = []
    code_files = []
    text_files = []
    
    for f in result.files:
        status = f.get("status", "unknown")
        if status == "extracted":
            files_summary.append(f"- {f['name']} ({f.get('size_formatted', '?')}) - {f.get('category', '?')}")
            if f.get("category") == "code":
                code_files.append(f)
            else:
                text_files.append(f)
        elif status in ("binary", "media"):
            files_summary.append(f"- {f['name']} ({f.get('size_formatted', '?')}) - {status}")
        elif status == "skipped":
            files_summary.append(f"- {f['name']} - skipped: {f.get('reason', '?')}")
        else:
            files_summary.append(f"- {f['name']} - {status}")
    
    summary_intro = f"""Archive Analysis: {result.metadata.get('filename', 'unknown')}
Total entries: {result.metadata.get('entry_count', 0)}
Processed: {result.metadata.get('processed_count', 0)}
Text files extracted: {result.metadata.get('extracted_count', 0)}

Files found:
{chr(10).join(files_summary)}

"""

    full_content = summary_intro + result.content
    
    messages = [
        {
            "role": "system",
            "content": get_system_prompt("") + """

You are analyzing an archive file (ZIP, TAR, etc.). The archive contents have been extracted and provided below.

Your task:
1. Provide an overview of what this archive contains
2. Identify the main purpose/type of the project or files
3. If it's a code project, describe the structure, technologies used, and main functionality
4. Highlight any important files or configurations
5. Note any potential issues, missing files, or areas of concern
6. If appropriate, provide a summary of the code functionality

Be organized and clear in your analysis."""
        },
        {
            "role": "user",
            "content": full_content
        }
    ]

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                # First, send metadata
                yield sse({
                    "type": "file_metadata",
                    "metadata": result.metadata,
                    "files": result.files
                })
                
                # Using Gemini for archive analysis as well
                async for token in stream_gemini_chat(messages):
                    if task.cancelled():
                        break
                    yield sse({"type": "token", "text": token})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Archive analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-stream fallback
    full_text = ""
    async for token in stream_gemini_chat(messages):
        full_text += token

    return {
        "analysis": full_text,
        "metadata": result.metadata,
        "files": result.files
    }

@app.get("/file-types")
async def get_supported_file_types():
    """Return list of supported file types"""
    return {
        "code": sorted(list(CODE_EXTENSIONS)),
        "document": sorted(list(DOCUMENT_EXTENSIONS)),
        "data": sorted(list(DATA_EXTENSIONS)),
        "image": sorted(list(IMAGE_EXTENSIONS)),
        "audio": sorted(list(AUDIO_EXTENSIONS)),
        "video": sorted(list(VIDEO_EXTENSIONS)),
        "archive": sorted(list(ARCHIVE_EXTENSIONS)),
        "limits": {
            "max_file_size": format_file_size(MAX_FILE_SIZE),
            "max_zip_size": format_file_size(MAX_ZIP_SIZE),
            "max_zip_entries": MAX_ZIP_ENTRIES,
            "max_extracted_size": format_file_size(MAX_EXTRACTED_SIZE),
            "max_text_length": format_file_size(MAX_TEXT_LENGTH)
        }
    }

# =========================
# SESSION MANAGEMENT ENDPOINTS
# =========================
@app.post("/session/validate")
async def validate_session(req: Request, res: Response):
    """Validate current session and return user info"""
    user = await get_user(req, res)
    return {
        "valid": user.get("session_valid", False),
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_authenticated": bool(user.get("email") and not user["email"].startswith("anon+"))
    }

@app.post("/session/refresh")
async def refresh_session(req: Request, res: Response):
    """Manually refresh the current session"""
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}
    remember = body.get("remember", True)
    
    user = await get_user(req, res, remember=remember)
    
    # Force session refresh
    new_token = await create_user_session(
        user["id"],
        user.get("fingerprint", ""),
        remember
    )
    
    set_session_cookies(res, user["id"], user.get("fingerprint", ""), new_token, remember)
    
    return {
        "status": "refreshed",
        "user_id": user["id"],
        "expires_in": SESSION_DURATION if remember else 24 * 60 * 60
    }

@app.post("/session/logout")
async def logout(req: Request, res: Response):
    """Logout and clear all session data"""
    user_id = req.cookies.get(PRIMARY_COOKIE)
    
    if user_id:
        try:
            # Invalidate all sessions for this user
            await _execute_supabase_with_retry(
                supabase.table("user_sessions")
                .update({"is_valid": False})
                .eq("user_id", user_id),
                description="Invalidate User Sessions"
            )
        except Exception as e:
            logger.error(f"Failed to invalidate sessions: {e}")
        
        # Clear cache
        if user_id in _session_cache:
            del _session_cache[user_id]
    
    clear_session_cookies(res)
    
    return {"status": "logged_out"}

# =========================
# CHAT MANAGEMENT ENDPOINTS
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    user = await get_user(req, res)
    cid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    
    await _execute_supabase_with_retry(
        supabase.table("conversations").insert({
            "id": cid, 
            "user_id": user["id"],
            "title": "New Chat", 
            "created_at": now_iso,
            "updated_at": now_iso
        }),
        description="New Chat"
    )
    return {"conversation_id": cid}

@app.get("/chat/{conversation_id}/messages")
async def get_chat_messages(conversation_id: str, req: Request, res: Response):
    """Fetches full history for a specific chat."""
    user = await get_user(req, res)
    
    # Verify the conversation belongs to the user
    conv_check = await _execute_supabase_with_retry(
        supabase.table("conversations").select("id").eq("id", conversation_id).eq("user_id", user["id"]).limit(1),
        description="Verify Chat Ownership"
    )
    
    if not conv_check.data:
        raise HTTPException(403, "Access denied to this conversation")

    # Fetch messages
    msgs = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False),
        description="Get Chat History"
    )
    
    return {"messages": msgs.data or []}

@app.post("/stop")
async def stop_generation(req: Request, res: Response):
    user = await get_user(req, res)
    user_id = user["id"]

    task = active_streams.get(user_id)
    if task and not task.done():
        task.cancel()
        active_streams.pop(user_id, None)
        return {"status": "stopped"}
    return {"status": "no_active_stream"}

@app.post("/regenerate")
async def regenerate(req: Request, res: Response):
    body = await req.json()
    conv_id = body.get("conversation_id")

    user = await get_user(req, res)
    user_id = user["id"]

    if not conv_id:
        raise HTTPException(400, "conversation_id required")

    msgs = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("*")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=True)
        .limit(10),
        description="Regenerate History Lookup"
    )

    if not msgs.data:
        raise HTTPException(404, "No messages found")

    last_user_msg = None
    for m in msgs.data:
        if m["role"] == "user":
            last_user_msg = m
            break

    if not last_user_msg:
        raise HTTPException(400, "No user message to regenerate from")

    await _execute_supabase_with_retry(
        supabase.table("messages")
        .delete()
        .gt("created_at", last_user_msg["created_at"])
        .eq("role", "assistant")
        .eq("conversation_id", conv_id),
        description="Delete Old Assistant Message"
    )

    async def event_gen():
        task = asyncio.current_task()
        active_streams[user_id] = task
        try:
            history = await get_history(conv_id)
            
            last_prompt = last_user_msg.get("content", "")
            base_system = get_system_prompt(last_prompt)
            user_memory = user.get("memory", "")
            if user_memory:
                base_system += f"\n\nUser Context: {user_memory}"
            full_history = [{"role": "system", "content": base_system}] + history
            
            full_text = ""
            # Using Gemini for regeneration
            async for token in stream_gemini_chat(full_history):
                if task and task.cancelled():
                    break
                full_text += token
                yield sse({"type": "token", "text": token})

            asyncio.create_task(update_user_memory(user["id"], user_memory, last_prompt, full_text))

            try:
                await save_message(user_id, conv_id, "assistant", full_text)
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")

            yield sse({"type": "done"})
        
        except Exception as e:
            logger.error(f"Regenerate Stream Error: {e}")
            yield sse({"type": "error", "message": "An error occurred."})
        
        finally:
            active_streams.pop(user_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/chats")
async def list_chats(req: Request, res: Response):
    """Returns a list of all conversations for the logged-in user, ordered by most recently active."""
    user = await get_user(req, res)
    
    result = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True),
        description="List Chats"
    )
    return {"chats": result.data or []}

# =========================
# USER IDENTITY ENDPOINT 
# =========================
@app.get("/user/info")
async def get_user_info(req: Request, res: Response):
    user = await get_user(req, res)
    return {
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_identified": True,
        "session_valid": user.get("session_valid", False),
        "is_authenticated": bool(user.get("email") and not user["email"].startswith("anon+"))
    }

@app.post("/user/merge")
async def merge_user(req: Request, res: Response):
    body = await req.json()
    target_id = body.get("target_user_id")
    
    user = await get_user(req, res)
    
    if not target_id or target_id == user["id"]:
        return {"status": "no_merge_needed"}
    
    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations")
            .update({"user_id": target_id})
            .eq("user_id", user["id"]),
            description="Merge Conversations"
        )
        
        await _execute_supabase_with_retry(
            supabase.table("messages")
            .update({"user_id": target_id})
            .eq("user_id", user["id"]),
            description="Merge Messages"
        )
        
        fingerprint = user.get("fingerprint", "")
        session_token = await create_user_session(target_id, fingerprint, True)
        set_session_cookies(res, target_id, fingerprint, session_token, True)
        
        return {"status": "merged", "new_user_id": target_id}
    
    except Exception as e:
        logger.error(f"User merge failed: {e}")
        raise HTTPException(500, "Failed to merge user data")

# =========================
# INTENT ANALYSIS ENDPOINT
# =========================
@app.post("/analyze-intent")
async def analyze_intent_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(400, "Prompt required")

    intent_result = detect_intent(prompt)
    action_type = get_action_type(prompt)
    required_tools = get_required_tools(prompt)

    return {
        "intent": intent_result.to_dict() if intent_result else None,
        "address: "action_type": action_type,
        "required_tools": required_tools,
        "confidence": intent_result.confidence if intent_result else 0.0
    }

# =========================
# MEDIA ENDPOINTS (OPTIMIZED FOR SPEED)
# =========================

@app.post("/tts")
async def text_to_speech(req: Request):
    """
    Optimized TTS: Streams audio back immediately as it is generated.
    This reduces latency significantly for long texts.
    """
    # Validate Content-Type
    content_type = req.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(415, "Unsupported Media Type: Expected application/json")

    try:
        data = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body or empty request body")

    text = data.get("text")
    voice = data.get("voice", "alloy")

    if not text:
        raise HTTPException(400, "text required")
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI Key missing")

    # Use streaming to get audio back faster
    async def stream_audio():
        async with httpx.AsyncClient(timeout=60.0) as client_http:
            async with client_http.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={"model": "tts-1", "voice": voice, "input": text, "response_format": "mp3"}
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"TTS Error: {error_body}")
                    return
                
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "echo", "name": "Echo"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"}
        ]
    }

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Fixed STT: Complete implementation with optimized httpx usage.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI Key missing")

    content = await file.read()
    
    # Optimized: Use a reasonable timeout and efficient async client
    async with httpx.AsyncClient(timeout=30.0) as client_http:
        files = {"file": (file.filename, content, file.content_type)}
        data = {"model": "whisper-1"}
        
        try:
            r = await client_http.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files, 
                data=data
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"STT Error: {e.response.text}")
            raise HTTPException(e.response.status_code, f"STT Failed: {e.response.text}")
        except Exception as e:
            logger.error(f"STT Exception: {e}")
            raise HTTPException(500, "Speech to Text failed")

# =========================
# STARTUP
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
