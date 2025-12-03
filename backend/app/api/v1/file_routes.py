from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Dict, Any
from app.services.file_service import file_service
from app.utils.dependencies import get_current_user
from app.models.user import TokenData

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload/image", response_model=Dict[str, Any])
async def upload_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    📷 Upload image with OCR text extraction.
    
    - Extracts text from image using Tesseract OCR
    - Saves file to user's folder
    - Max size: 10MB
    """
    try:
        result = await file_service.process_image(file, current_user.user_id)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Image upload failed: {str(e)}")


@router.post("/upload/pdf", response_model=Dict[str, Any])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    📄 Upload PDF with text extraction.
    
    - Extracts all text from PDF pages
    - Returns page count
    - Max size: 10MB
    """
    try:
        result = await file_service.process_pdf(file, current_user.user_id)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"PDF upload failed: {str(e)}")


@router.post("/upload/document", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(..., description="DOCX or TXT file"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    📝 Upload DOCX or TXT document.
    
    - Extracts text content
    - Supports DOCX and TXT formats
    - Max size: 10MB
    """
    try:
        if file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            result = await file_service.process_docx(file, current_user.user_id)
        elif file.content_type == "text/plain":
            result = await file_service.process_text_file(file, current_user.user_id)
        else:
            raise HTTPException(400, "Unsupported document type. Use DOCX or TXT.")
        
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Document upload failed: {str(e)}")


@router.post("/upload/audio", response_model=Dict[str, Any])
async def upload_audio(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, WEBM, OGG, M4A)"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    🎤 Transcribe audio to text using OpenAI Whisper.
    
    - Supports MP3, WAV, WEBM, OGG, M4A
    - Returns full transcription
    - Max size: 10MB
    - Requires OPENAI_API_KEY in environment
    """
    try:
        result = await file_service.transcribe_audio(file, current_user.user_id)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Audio transcription failed: {str(e)}")


@router.delete("/delete")
async def delete_file(
    file_path: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    🗑️ Delete uploaded file.
    
    - Requires file_path (relative path from upload dir)
    - User can only delete their own files
    """
    try:
        success = file_service.delete_file(file_path, current_user.user_id)
        if not success:
            raise HTTPException(404, "File not found or access denied")
        return {"success": True, "message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"File deletion failed: {str(e)}")


@router.get("/health")
async def file_service_health():
    """🏥 Health check for file service"""
    return {
        "status": "healthy",
        "service": "file_upload",
        "supported_formats": {
            "images": ["JPG", "PNG", "WEBP"],
            "documents": ["PDF", "DOCX", "TXT"],
            "audio": ["MP3", "WAV", "WEBM", "OGG", "M4A"]
        }
    }