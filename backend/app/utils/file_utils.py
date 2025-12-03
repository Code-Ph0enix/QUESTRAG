import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import uuid

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_DOC_TYPES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/webm", "audio/ogg", "audio/m4a"}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file_type(file: UploadFile, allowed_types: set) -> bool:
    """Check if file type is allowed"""
    return file.content_type in allowed_types

def validate_file_size(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bool:
    """Check if file size is under limit"""
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    return size <= max_size

def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename with UUID"""
    ext = Path(original_filename).suffix
    return f"{uuid.uuid4()}{ext}"

async def save_upload_file(file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination