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
# Using OpenRouter for Llama 8B
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
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
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for this backend.")

# =========================
# LIFESPAN EVENT HANDLER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("HeloxAi Backend Started. Model: Llama 8B via OpenRouter.")
    yield
    # Shutdown
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
_session_cache_ttl = 300  # 5 minutes
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

# Comprehensive file type mappings
CODE_EXTENSIONS = {
    '.py', '.pyw', '.pyx', '.pyd', '.pyi', '.py3',
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte', '.astro',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.clj', '.cljs', '.hs',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
    '.cs', '.csx',
    '.go',
    '.rs',
    '.php', '.phtml',
    '.rb', '.erb', '.rake', '.gemspec',
    '.swift',
    '.dart',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    '.lua',
    '.pl', '.pm',
    '.r', '.R',
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
        '.html': 'html', '.htm': 'html',
        '.css': 'css', '.scss': 'scss', '.less': 'less',
        '.vue': 'vue', '.svelte': 'svelte',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
        '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.dart': 'dart',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
        '.ps1': 'powershell', '.bat': 'batch',
        '.lua': 'lua',
        '.pl': 'perl',
        '.r': 'r', '.R': 'r',
        '.sql': 'sql',
        '.json': 'json', '.xml': 'xml',
        '.yaml': 'yaml', '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown', '.rst': 'rst',
        '.tex': 'latex',
        '.dockerfile': 'dockerfile',
        '.graphql': 'graphql', '.gql': 'graphql',
        '.tf': 'hcl', '.hcl': 'hcl',
        '.sol': 'solidity',
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)

def is_binary_file(filename: str, content: bytes = None) -> bool:
    """Check if file is binary based on extension or content"""
    ext = Path(filename).suffix.lower()
    
    binary_exts = {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
        '.pyc', '.pyo', '.class', '.o', '.obj', '.a', '.lib',
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.sqlite', '.db', '.sqlite3',
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.pak', '.bundle',
    }
    
    if ext in binary_exts:
        return True
    
    if content and len(content) > 0:
        check_bytes = content[:8192]
        if b'\x00' in check_bytes:
            return True
    
    return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
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
        "category": category.value,
        "size": original_size,
        "size_formatted": format_file_size(original_size),
        "language": get_file_language(filename),
    }

    try:
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)

        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)

        if category in (FileCategory.CODE, FileCategory.CONFIG, FileCategory.UNKNOWN):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=original_size
            )

        if category in (FileCategory.DOCUMENT, FileCategory.DATA):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=original_size
            )

        if is_binary_file(filename, content):
            return FileExtractionResult(
                content=f"[Binary file: {filename} ({format_file_size(original_size)}) - Cannot extract text content]",
                metadata=metadata,
                original_size=original_size
            )

        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(
            content=text,
            metadata=metadata,
            truncated=truncated,
            original_size=original_size
        )

    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return FileExtractionResult(
            content=f"[Error extracting {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=original_size
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
    """Extract and read contents from archive files (zip, tar, etc.)"""
    ext = Path(filename).suffix.lower()
    
    if ext == '.zip':
        return await extract_zip_content(content, filename, max_length, metadata)
    elif ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
        return await extract_tar_content(content, filename, max_length, metadata)
    elif ext in ('.7z', '.rar'):
        return FileExtractionResult(
            content=f"[{ext.upper()} archive: {filename} ({format_file_size(len(content))}) - This archive format requires additional server setup]",
            metadata=metadata,
            original_size=len(content)
        )
    else:
        return FileExtractionResult(
            content=f"[Archive: {filename} ({format_file_size(len(content))}) - Unsupported archive format]",
            metadata=metadata,
            original_size=len(content)
        )

async def extract_zip_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract and read text contents from ZIP files"""
    extracted_files = []
    all_text_parts = []
    total_extracted = 0
    entry_count = 0
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[ZIP archive: {filename} - Too many entries ({len(zf.namelist())}). Maximum allowed: {MAX_ZIP_ENTRIES}]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            entries = sorted(zf.namelist())
            
            for entry_name in entries:
                if entry_name.endswith('/') or '/__MACOSX/' in entry_name:
                    continue
                if entry_name.startswith('__MACOSX') or entry_name.startswith('.'):
                    continue
                
                entry_count += 1
                
                try:
                    entry_info = zf.getinfo(entry_name)
                    
                    if entry_info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": f"File too large (max {format_file_size(MAX_FILE_SIZE)})"
                        })
                        continue
                    
                    if total_extracted + entry_info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": "Archive total size limit reached"
                        })
                        continue
                    
                    entry_content = zf.read(entry_name)
                    total_extracted += len(entry_content)
                    
                    entry_category = get_file_category(entry_name)
                    entry_language = get_file_language(entry_name)
                    
                    if is_binary_file(entry_name, entry_content):
                        file_info = {
                            "name": entry_name,
                            "size": len(entry_content),
                            "size_formatted": format_file_size(len(entry_content)),
                            "category": "binary",
                            "status": "binary",
                            "note": "Binary file - cannot extract text"
                        }
                    else:
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        
                        if text.strip():
                            file_info = {
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "language": entry_language,
                                "status": "extracted",
                                "line_count": text.count('\n') + 1,
                                "preview": text[:500] + ("..." if len(text) > 500 else "")
                            }
                            all_text_parts.append(f"\n{'='*60}\nFile: {entry_name}\n{'='*60}\n{text}")
                        else:
                            file_info = {
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "status": "empty",
                                "note": "File is empty"
                            }
                    
                    extracted_files.append(file_info)
                    
                except Exception as e:
                    extracted_files.append({
                        "name": entry_name,
                        "status": "error",
                        "error": str(e)
                    })
        
        full_text = f"ZIP Archive: {filename}\n"
        full_text += f"Total entries: {len(zf.namelist())}, Processed: {entry_count}\n"
        full_text += f"Extracted text files: {len(all_text_parts)}\n"
        full_text += f"Total extracted size: {format_file_size(total_extracted)}\n\n"
        
        if all_text_parts:
            full_text += "=".join(["="*30]) + "\n"
            full_text += "EXTRACTED CONTENT\n"
            full_text += "=".join(["="*30])
            full_text += "".join(all_text_parts)
        else:
            full_text += "No text content could be extracted from this archive.\n\n"
            full_text += "Files found:\n"
            for f in extracted_files:
                status = f.get('status', 'unknown')
                full_text += f"  - {f['name']} ({f.get('size_formatted', '?')}) [{status}]\n"
        
        metadata.update({
            "archive_type": "zip",
            "entry_count": len(zf.namelist()),
            "processed_count": entry_count,
            "extracted_count": len(all_text_parts),
            "total_extracted_size": total_extracted,
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
        
    except zipfile.BadZipFile:
        return FileExtractionResult(
            content=f"[Error: {filename} is not a valid ZIP file or is corrupted]",
            metadata=metadata,
            original_size=len(content)
        )
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return FileExtractionResult(
            content=f"[Error extracting ZIP {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )

async def extract_tar_content(
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
                    content=f"[TAR archive: {filename} - Too many entries ({len(members)})]",
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
                    if f is None:
                        continue
                    
                    entry_content = f.read()
                    entry_category = get_file_category(member.name)
                    
                    if not is_binary_file(member.name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {member.name}\n{'='*60}\n{text}")
                            extracted_files.append({
                                "name": member.name,
                                "size": member.size,
                                "status": "extracted",
                                "category": entry_category.value
                            })
                    else:
                        extracted_files.append({
                            "name": member.name,
                            "size": member.size,
                            "status": "binary",
                            "category": entry_category.value
                        })
                        
                except Exception as e:
                    extracted_files.append({
                        "name": member.name,
                        "status": "error",
                        "error": str(e)
                    })
        
        full_text = f"TAR Archive: {filename}\n"
        full_text += f"Entries: {len(members)}, Extracted: {len(all_text_parts)}\n\n"
        
        if all_text_parts:
            full_text += "".join(all_text_parts)
        
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
            content=f"[Error extracting TAR {filename}: {str(e)}]",
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
        DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE
    ]
    
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    
    for cookie_name in cookies_to_clear:
        delete_kwargs = {
            "key": cookie_name,
            "path": "/",
            "secure": True,
            "samesite": "none"
        }
        if cookie_domain:
            delete_kwargs["domain"] = cookie_domain
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
# BASE SYSTEM PROMPT (UPDATED FOR LLAMA 8B)
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
    r'\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b.*\b(is|are|who)\b',
    r'\bwho\b.*\bbehind\b.*\b(you|this|helox)\b',
    r'\bwho.*made.*you\b',
    r'\bwho.*created.*you\b',
    r'\bwho.*built.*you\b',
    r'\bwho.*developed.*you\b',
    r'\bwho.*programmed.*you\b',
    r'\bwho.*constructed.*you\b',
    r'\bwho.*designed.*you\b',
    r'\bwho.*owns.*you\b',
    r'\bwho.*runs.*you\b',
    r'\byour\s+creator\b',
    r'\byour\s+developer\b',
    r'\byour\s+maker\b',
    r'\byour\s+builder\b',
    r'\byour\s+founder\b',
    r'\byour\s+owner\b',
    r'\bwho\s+is\s+behind\s+helox\b',
    r'\bwho\s+made\s+helox\b',
    r'\bwho\s+created\s+helox\b',
    r'\bwho\s+built\s+helox\b',
    r'\bwho\s+developed\s+helox\b',
    r'\bmade\s+by\s+who\b',
    r'\bcreated\s+by\s+who\b',
    r'\bbuilt\s+by\s+who\b',
    r'\bdeveloped\s+by\s+who\b',
    r'\bconstructed\s+by\s+who\b',
    r'\btell\s+me\s+about\s+your\s+(creator|developer|maker|builder|founder)\b',
    r'\bwhat\s+company\s+made\s+you\b',
    r'\bwhat\s+team\s+made\s+you\b',
    r'\bwhere\s+do\s+you\s+come\s+from\b',
    r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
    r'\bare\s+you\s+made\s+by\b',
    r'\bdid\s+.*\s+make\s+you\b',
    r'\bdid\s+.*\s+create\s+you\b',
    r'\bdid\s+.*\s+build\s+you\b',
]

COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]

def is_creator_question(text: str) -> bool:
    """Check if user is asking about who created/made the AI"""
    for pattern in COMPILED_CREATOR_PATTERNS:
        if pattern.search(text):
            return True
    return False

def get_system_prompt(user_prompt: str) -> str:
    """Return base prompt normally, creator response ONLY if asked"""
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT

# =========================
# ADVANCED INTENT DETECTION (UPDATED)
# =========================
class IntentCategory(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    DOCUMENT_CREATION = "document_creation"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    WEB_DEVELOPMENT = "web_development"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    CONVERSATION = "conversation"

@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    sub_intents: List[IntentCategory]
    keywords_matched: List[str]
    patterns_matched: List[str]

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence,3),
            "sub_intents": [i.value for i in self.sub_intents],
            "keywords_matched": self.keywords_matched,
            "patterns_matched": self.patterns_matched
        }

class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {
            "don't", "dont", "do not", "doesn't", "doesnt", "does not",
            "didn't", "didnt", "did not", "never", "no", "not", "without",
            "skip", "avoid", "except", "but not", "ignore", "rather than"
        }

    def _compile_patterns(self):
        self.patterns = {
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)',
                r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))',
                r'\b(code\s+(for|that|this|to|which|example))',
                r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))',
                r'\b(scaffold|boilerplate|template)\s+(for|a)',
                r'\b(wrapper|helper|utility)\s+(function|class|module)\s+(for|to)',
                r'\b(implement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature))',
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)',
                r'\b(is\s+(this|there)\s+(code|anything)\s+(good|bad|wrong|improvable|clean))',
                r'\b(best\s+practices?\s+(for|in)\s+(this|my)\s+(code|implementation))',
                r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)',
                r'\b(code\s+quality|technical\s+debt|code\s+smell)',
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)',
                r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring|returning))',
                r'\b(error|exception|traceback|stack\s+trace|segfault)\s*[:\n]',
                r'\b(what(\'s|\s+is)\s+(wrong|the\s+problem)\s+(with|in))',
                r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)',
                r'\b(unexpected|wrong|incorrect)\s+(result|output|behavior|value)',
                r'\b(help\s+(me\s+)?)?debug',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
                r'\b(format\s+(as|this\s+as|it\s+as)\s+(a\s+)?(pdf|document|report|letter|markdown))',
                r'\b(professional|formal|business)\s+(document|letter|email|report)',
            ],
            IntentCategory.DATA_ANALYSIS: [
                r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)',
                r'\b(statistics?|statistical)\s+(analysis|test|summary|overview)',
                r'\b(insights?\s+(from|in|about|into))',
                r'\b(correlation|regression|distribution|trend)\s+(analysis|of|in)',
                r'\b(clean|preprocess|prepare|wrangle)\s+(this|the)\s+(data|dataset)',
                r'\b(eda|exploratory\s+data\s+analysis)',
            ],
            IntentCategory.DATA_VISUALIZATION: [
                r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)',
                r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap|box\s+plot|violin\s+plot)',
                r'\b(visualize|visualise|plot|chart|graph)\s+(this|the|these|those|data)',
                r'\b(matplotlib|seaborn|plotly|d3|chart\.js|ggplot|altair)',
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page|web\s*site|portfolio)',
                r'\b(html|css|javascript|typescript|react|vue|angular|next\.js|nuxt|svelte|tailwind)\b',
                r'\b(frontend|front[- ]end|back[- ]end|full[- ]stack)\s*(development|for|with|app)?',
                r'\b(responsive|mobile[- ]friendly|mobile[- ]first)\s*(design|website|layout)?',
                r'\b(component|page|layout|template)\s+(for|in)\s+(react|vue|angular|next)',
            ],
            IntentCategory.API_DEVELOPMENT: [
                r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)',
                r'\b(api\s*(endpoint|route|handler|controller|gateway))',
                r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service|endpoint)?',
                r'\b(openapi|swagger|api\s*documentation)',
                r'\b(request|response|payload)\s+(format|structure|schema)',
            ],
            IntentCategory.DATABASE: [
                r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)',
                r'\b(sql|mysql|postgres|postgresql|mongodb|redis|dynamodb|sqlite)\s*(query|statement|command)?',
                r'\b(schema\s*(design|migration|definition|update))',
                r'\b(orm|sequelize|prisma|sqlalchemy|typeorm|drizzle)\s*(query|model|schema)?',
                r'\b(crud\s*(operation|operations|endpoint|api))',
                r'\b(select|insert|update|delete)\s+(from|into|table)',
            ],
            IntentCategory.TRANSLATION: [
                r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)',
                r'\b(in|to|into)\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu)',
                r'\b(how\s+(do\s+you|to)\s+say\s+.+\s+in\s+\w+)',
                r'\b(native|localize|localization|l10n|i18n|internationaliz)',
            ],
            IntentCategory.SUMMARIZATION: [
                r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)',
                r'\b(brief|short|concise)\s+(overview|summary|explanation|version)\s*(of|for|about)?',
                r'\b(key\s+(points|takeaways|highlights))\s*(from|of|in)?',
                r'\b(main\s+(idea|points|theme|argument|concept))',
                r'\b(give\s+me\s+(the\s+)?(gist|bottom\s+line|essence))',
            ],
            IntentCategory.EXPLANATION: [
                r'\b(explain|explanation)\s+(to\s+me\s+)?',
                r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+',
                r'\b(how\s+(does|do|did|can|would|should|to))\s+',
                r'\b(tell\s+me\s+(about|more\s+about|how|why))',
                r'\b(why\s+(is|does|do|are|did|can|would))\s+',
                r'\b(definition|meaning)\s+(of|for)\s+',
                r'\b(understand(ing)?)\s*(this|how|why|what|better)?',
                r'\b(break\s+down|simplify|elaborate)\s+',
            ],
            IntentCategory.CREATIVE_WRITING: [
                r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|song|haiku|limerick)',
                r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller|mystery)\s*(writing|story|tale)?',
                r'\b(narrative|plot|character|setting|dialogue)\s*(for|development|creation|arc)?',
                r'\b(storytelling|story[- ]?telling)',
                r'\b(write\s+(like|in\s+the\s+style\s+of))\s+',
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem|integral|derivative)?',
                r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability|linear\s+algebra)\s*(problem|equation|question)?',
                r'\b(\d+[\.\d]*\s*[\+\-\*\/\^%\=]\s*[\.\d]*)',
                r'\b(integral|derivative|differentiat|integrat)\s*(of|the)?',
                r'\b(prove|proof)\s+(that|this|the)',
                r'\b(formula|equation)\s+(for|to\s+calculate|to\s+find)',
            ],
            IntentCategory.RESEARCH: [
                r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)',
                r'\b(stud(y|ies))\s+(show|suggest|indicate|demonstrate|prove)',
                r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper|article|research|journal)?',
                r'\b(cite|citation|reference|bibliography)\s+',
                r'\b(literature\s+review)\s*(on|for|of)?',
                r'\b(what\s+(does\s+)?(research|science|literature)\s+say)',
                r'\b(latest\s+news|current\s+events|what\s+is\s+happening)',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$',
                r'^(how\s+are\s+you|how(\'s|\s+is)\s+it\s+going|what(\'s|\s+is)\s+up)[\s!.?]*$',
                r'^(bye|goodbye|see\s+you|farewell)[\s!.?]*$',
                r'^(sure|okay|ok|got\s+it|understood)[\s!.?]*$',
            ],
        }

        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.CODE_GENERATION: [
                "function", "class", "method", "script", "program", "code",
                "algorithm", "implementation", "module", "component", "app",
                "application", "utility", "helper", "wrapper", "snippet"
            ],
            IntentCategory.CODE_REVIEW: [
                "review", "analyze", "critique", "evaluate", "audit",
                "refactor", "improve", "optimize", "clean", "quality"
            ],
            IntentCategory.CODE_DEBUG: [
                "bug", "error", "issue", "problem", "fix", "debug",
                "troubleshoot", "resolve", "not working", "broken", "failing"
            ],
            IntentCategory.DOCUMENT_CREATION: [
                "document", "pdf", "report", "letter", "email", "memo",
                "article", "essay", "paper", "proposal", "whitepaper"
            ],
            IntentCategory.DATA_ANALYSIS: [
                "analyze", "analysis", "data", "dataset", "statistics",
                "insights", "correlation", "regression", "distribution"
            ],
            IntentCategory.DATA_VISUALIZATION: [
                "chart", "graph", "plot", "visualization", "diagram",
                "dashboard", "matplotlib", "seaborn", "plotly"
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                "website", "webpage", "web app", "landing page", "html",
                "css", "javascript", "react", "vue", "angular", "next.js"
            ],
            IntentCategory.API_DEVELOPMENT: [
                "api", "rest", "graphql", "endpoint", "route", "swagger",
                "openapi", "request", "response"
            ],
            IntentCategory.DATABASE: [
                "database", "schema", "table", "query", "sql", "orm",
                "migration", "select", "insert", "update", "delete"
            ],
            IntentCategory.TRANSLATION: [
                "translate", "translation", "language", "localize",
                "internationalize", "i18n", "l10n"
            ],
            IntentCategory.SUMMARIZATION: [
                "summarize", "summary", "tldr", "brief", "overview",
                "key points", "takeaways", "highlights"
            ],
            IntentCategory.EXPLANATION: [
                "explain", "explanation", "what is", "how does", "why",
                "definition", "meaning", "understand", "simplify"
            ],
            IntentCategory.CREATIVE_WRITING: [
                "story", "poem", "poetry", "novel", "chapter", "verse",
                "lyrics", "song", "haiku", "creative", "fiction"
            ],
            IntentCategory.MATHEMATICAL: [
                "calculate", "compute", "solve", "equation", "formula",
                "math", "algebra", "calculus", "geometry", "statistics"
            ],
            IntentCategory.RESEARCH: [
                "research", "find", "search", "investigate", "study",
                "academic", "scholarly", "citation", "reference"
            ],
            IntentCategory.CONVERSATION: [
                "hello", "hi", "hey", "greetings", "thanks", "thank you",
                "goodbye", "bye", "okay", "ok"
            ],
        }

    def detect_intent(self, text: str) -> IntentResult:
        """Detect the primary intent from user input"""
        if not text or not text.strip():
            return IntentResult(
                intent=IntentCategory.CONVERSATION,
                confidence=0.5,
                sub_intents=[],
                keywords_matched=[],
                patterns_matched=[]
            )

        text_lower = text.lower()
        intent_scores = {}
        pattern_matches = {}
        keyword_matches = {}

        # Check for negation
        has_negation = any(neg in text_lower for neg in self.negation_words)

        # Pattern matching
        for intent, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern.search(text):
                    matches.append(pattern.pattern)
            
            if matches:
                pattern_matches[intent] = matches
                intent_scores[intent] = intent_scores.get(intent, 0) + (len(matches) * 0.3)

        # Keyword matching
        for intent, keywords in self.synonyms.items():
            matches = [kw for kw in keywords if kw in text_lower]
            
            if matches:
                keyword_matches[intent] = matches
                intent_scores[intent] = intent_scores.get(intent, 0) + (len(matches) * 0.1)

        # Apply negation penalty
        if has_negation:
            for intent in list(intent_scores.keys()):
                intent_scores[intent] *= 0.3

        if not intent_scores:
            return IntentResult(
                intent=IntentCategory.CONVERSATION,
                confidence=0.5,
                sub_intents=[],
                keywords_matched=[],
                patterns_matched=[]
            )

        # Normalize scores
        max_score = max(intent_scores.values())
        if max_score > 0:
            intent_scores = {k: v / max_score for k, v in intent_scores.items()}

        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        confidence = intent_scores[primary_intent]

        # Get sub-intents (lower confidence)
        sub_intents = [
            intent for intent, score in intent_scores.items()
            if intent != primary_intent and score >= 0.3
        ]

        return IntentResult(
            intent=primary_intent,
            confidence=confidence,
            sub_intents=sub_intents,
            keywords_matched=keyword_matches.get(primary_intent, []),
            patterns_matched=pattern_matches.get(primary_intent, [])
        )

# Initialize intent detector
intent_detector = AdvancedIntentDetector()

# =========================
# HELPER FUNCTIONS
# =========================
async def _execute_supabase_with_retry(operation, description: str = "Supabase Operation", max_retries: int = 3):
    """Execute Supabase operation with retry logic"""
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

# =========================
# LLAMA 8B CHAT COMPLETION
# =========================
async def chat_completion_with_llama_8b(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> Union[Dict, AsyncGenerator]:
    """
    Call Llama 8B via OpenRouter API
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://heloxai.xyz",
        "X-Title": "HeloXAi"
    }
    
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",  # Llama 8B model
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
                        yield f"data: {{'error': 'API request failed'}}\n\n"
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
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter API error: {response.text}"
                )
            
            return response.json()

# =========================
# TAVILY WEB SEARCH
# =========================
async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API
    """
    if not TAVILY_API_KEY:
        return []
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }
    
    payload = {
        "query": query,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logger.error(f"Tavily API error: {response.status_code}")
                return []
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

# =========================
# MAIN CHAT ENDPOINT
# =========================
@app.post("/api/chat")
async def chat_endpoint(
    request: Request,
    response: Response
):
    """
    Main chat endpoint using Llama 8B with optional web search
    """
    try:
        data = await request.json()
        user_message = data.get("message", "")
        conversation_history = data.get("history", [])
        use_search = data.get("use_search", False)
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Detect intent
        intent_result = intent_detector.detect_intent(user_message)
        
        # Prepare system prompt
        system_prompt = get_system_prompt(user_message)
        
        # Perform web search if needed
        search_results = []
        if use_search or intent_result.intent == IntentCategory.RESEARCH:
            search_results = await search_web(user_message)
            
            if search_results:
                # Add search results to system prompt
                search_context = "\n\n**Web Search Results:**\n"
                for i, result in enumerate(search_results[:3], 1):
                    search_context += f"{i}. [{result.get('title', 'Untitled')}]({result.get('url', '')})\n"
                    search_context += f"   {result.get('content', '')[:200]}...\n\n"
                
                system_prompt += search_context
        
        # Prepare messages for Llama 8B
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            if msg.get("role") in ["user", "assistant"]:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Llama 8B with streaming
        stream_generator = await chat_completion_with_llama_8b(
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )
        
        # Return streaming response
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
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# FILE PROCESSING ENDPOINT
# =========================
@app.post("/api/process-file")
async def process_file_endpoint(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Process uploaded file and extract text content
    """
    try:
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {format_file_size(MAX_FILE_SIZE)}"
            )
        
        # Extract content
        result = await extract_file_content(content, file.filename)
        
        return JSONResponse(content=result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# API ENDPOINTS
# =========================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: str = "helox"
    mode: str = "general"
    files: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, str]]] = None

class NewChatRequest(BaseModel):
    model: str = "helox"

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


async def get_user_from_request(request: Request) -> Optional[Dict]:
    """Extract user from cookies or Authorization header"""
    # Try Authorization header first (Supabase JWT)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            # Verify with Supabase
            result = supabase.auth.get_user(token)
            if result and result.user:
                return {
                    "id": result.user.id,
                    "email": result.user.email,
                    "token": token,
                    "is_authed": True
                }
        except Exception as e:
            logger.debug(f"JWT validation failed: {e}")
    
    # Try cookies
    user_id = request.cookies.get(PRIMARY_COOKIE)
    session_token = request.cookies.get(SESSION_TOKEN_COOKIE)
    expiry_str = request.cookies.get(SESSION_EXPIRY_COOKIE)
    
    if user_id and session_token:
        if expiry_str and not is_session_expired(expiry_str):
            # Validate token
            if await validate_session_token(user_id, session_token):
                return {
                    "id": user_id,
                    "token": session_token,
                    "is_authed": True
                }
    
    # Guest user
    guest_id = request.headers.get("x-guest-id")
    if not guest_id:
        guest_id = request.cookies.get(BACKUP_COOKIE) or str(uuid.uuid4())
    
    return {
        "id": guest_id,
        "token": None,
        "is_authed": False
    }


async def stream_openrouter_response(
    messages: List[Dict[str, str]],
    system_prompt: str
) -> AsyncGenerator[str, None]:
    """Stream response from OpenRouter API"""
    
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://heloxai.xyz",
                "X-Title": "HeloXAi"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": full_messages,
                "stream": True,
                "max_tokens": 4096,
                "temperature": 0.7
            }
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"OpenRouter error {response.status_code}: {error_text}")
                yield f"[Error from AI provider: {response.status_code}]"
                return
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                if line.startswith("data: "):
                    data = line[6:]
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


async def search_web(query: str) -> List[Dict]:
    """Search the web using Tavily API"""
    if not TAVILY_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "api_key": TAVILY_API_KEY,
                    "max_results": 5,
                    "include_answer": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
    except Exception as e:
        logger.error(f"Web search error: {e}")
    
    return []


@app.post("/newchat")
async def new_chat(
    request: Request,
    response: Response,
    body: Optional[NewChatRequest] = None
):
    """Create a new chat session"""
    user = await get_user_from_request(request)
    chat_id = str(uuid.uuid4())
    
    # Set session cookies for authed users
    if user.get("is_authed"):
        fingerprint = request.cookies.get(FINGERPRINT_COOKIE) or generate_device_fingerprint(request)
        session_token = await create_user_session(user["id"], fingerprint)
        set_session_cookies(response, user["id"], fingerprint, session_token)
    
    return JSONResponse({
        "conversation_id": chat_id,
        "status": "ok"
    })


@app.post("/ask/universal")
async def ask_universal(
    request: Request,
    response: Response,
    body: ChatRequest
):
    """Main chat endpoint - handles all message types"""
    user = await get_user_from_request(request)
    
    # Build message history
    messages = []
    if body.history:
        messages = body.history.copy()
    
    # Add current message
    messages.append({"role": "user", "content": body.message})
    
    # Handle file content if present
    if body.files:
        file_context = "\n\n--- Attached Files ---\n"
        for f in body.files:
            if f.get("content"):
                file_context += f"\n**{f.get('name', 'File')}**:\n```\n{f['content']}\n```\n"
        messages[-1]["content"] += file_context
    
    # Determine if web search is needed
    system_prompt = get_system_prompt(body.message)
    
    if body.mode == "search" or any(kw in body.message.lower() for kw in ["latest", "current", "news", "today", "recent", "2024", "2025"]):
        search_results = await search_web(body.message)
        if search_results:
            search_context = "\n\n--- Web Search Results ---\n"
            for i, result in enumerate(search_results, 1):
                search_context += f"\n{i}. **{result.get('title', 'Untitled')}**\n"
                search_context += f"   URL: {result.get('url', '')}\n"
                search_context += f"   {result.get('content', '')[:500]}\n"
            search_context += "\n--- End Search Results ---\n"
            messages[-1]["content"] += search_context
            system_prompt += "\n\nYou have been provided with web search results. Use them to answer the user's question accurately. Cite sources by URL when possible."
    
    # Get intent for potential future use
    intent_detector = AdvancedIntentDetector()
    intent = intent_detector.detect_intent(body.message)
    
    # Stream the response
    async def generate():
        try:
            async for chunk in stream_openrouter_response(messages, system_prompt):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/ask/universal/sync")
async def ask_universal_sync(
    request: Request,
    response: Response,
    body: ChatRequest
):
    """Non-streaming version for fallback"""
    user = await get_user_from_request(request)
    
    messages = body.history.copy() if body.history else []
    messages.append({"role": "user", "content": body.message})
    
    if body.files:
        file_context = "\n\n--- Attached Files ---\n"
        for f in body.files:
            if f.get("content"):
                file_context += f"\n**{f.get('name', 'File')}**:\n```\n{f['content']}\n```\n"
        messages[-1]["content"] += file_context
    
    system_prompt = get_system_prompt(body.message)
    
    if body.mode == "search":
        search_results = await search_web(body.message)
        if search_results:
            search_context = "\n\n--- Web Search Results ---\n"
            for i, result in enumerate(search_results, 1):
                search_context += f"\n{i}. **{result.get('title', 'Untitled')}**\n"
                search_context += f"   URL: {result.get('url', '')}\n"
                search_context += f"   {result.get('content', '')[:500]}\n"
            messages[-1]["content"] += search_context
            system_prompt += "\n\nUse the web search results to answer. Cite sources."
    
    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://heloxai.xyz",
                    "X-Title": "HeloXAi"
                },
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [{"role": "system", "content": system_prompt}] + messages,
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                full_response = data["choices"][0]["message"]["content"]
            else:
                full_response = f"[Error: AI provider returned {resp.status_code}]"
    except Exception as e:
        full_response = f"[Error: {str(e)}]"
    
    return JSONResponse({
        "content": full_response,
        "conversation_id": body.conversation_id
    })


@app.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    request: Request
):
    """Delete a chat and its messages"""
    user = await get_user_from_request(request)
    
    # Delete from database if user is authed
    if user.get("is_authed") and sb:
        try:
            supabase.table("messages").delete().eq("conversation_id", chat_id).execute()
            supabase.table("conversations").delete().eq("id", chat_id).execute()
        except Exception as e:
            logger.warning(f"DB delete failed: {e}")
    
    return JSONResponse({"status": "ok"})


@app.post("/stt")
async def speech_to_text(
    request: Request,
    file: UploadFile = File(...)
):
    """Convert speech to text using OpenRouter's Whisper or fallback"""
    user = await get_user_from_request(request)
    
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    audio_data = await file.read()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        # Try using OpenAI-compatible Whisper via OpenRouter or direct API
        # For now, return a placeholder - implement with your preferred STT service
        async with httpx.AsyncClient(timeout=30.0) as client:
            # This would need a real STT endpoint - using OpenAI as example
            # You may need to use a different service like Deepgram, AssemblyAI, etc.
            return JSONResponse({
                "text": "[Speech-to-text not configured. Please set up a STT service.]",
                "status": "error"
            })
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/tts")
async def text_to_speech(
    request: Request,
    body: TTSRequest
):
    """Convert text to speech"""
    user = await get_user_from_request(request)
    
    if not user.get("is_authed"):
        raise HTTPException(status_code=401, detail="Sign in required for TTS")
    
    text = body.text[:4000]  # Limit text length
    
    try:
        # Using OpenAI TTS API as example - adjust for your provider
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": body.voice or "alloy"
                }
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    response.aiter_bytes(chunk_size=8192),
                    media_type="audio/mpeg"
                )
            else:
                raise HTTPException(status_code=500, detail="TTS generation failed")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...)
):
    """Upload and extract content from a file"""
    user = await get_user_from_request(request)
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {format_file_size(MAX_FILE_SIZE)}"
        )
    
    result = await extract_file_content(content, file.filename)
    
    return JSONResponse(result.to_dict())


@app.get("/user/plan")
async def get_user_plan(request: Request):
    """Get user's subscription plan (for frontend fallback)"""
    user = await get_user_from_request(request)
    
    if not user.get("is_authed"):
        return JSONResponse({"plan": "free"})
    
    try:
        result = supabase.table("users").select("plan, is_premium, is_lifetime").eq("id", user["id"]).maybe_single().execute()
        
        if result.data:
            data = result.data
            if data.get("is_lifetime"):
                return JSONResponse({"plan": "lifetime"})
            elif data.get("is_premium"):
                plan = data.get("plan", "ultimate_monthly")
                return JSONResponse({"plan": plan})
        
        return JSONResponse({"plan": "free"})
    except Exception as e:
        logger.error(f"Plan fetch error: {e}")
        return JSONResponse({"plan": "free"})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "model": "Llama 8B via OpenRouter",
        "version": "3.0.3",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# Catch-all for undefined routes (helps debugging)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Log and return 404 for undefined routes"""
    logger.warning(f"Undefined route: {request.method} /{path}")
    raise HTTPException(
        status_code=404,
        detail=f"Route not found: {request.method} /{path}"
    )
    
# =========================
# HEALTH CHECK ENDPOINT
# =========================
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model": "Llama 8B",
        "version": "3.0.3",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
