"""File upload security validation."""

import magic
import hashlib
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status
import logging

logger = logging.getLogger(__name__)

class FileSecurityValidator:
    """Validates uploaded files for security threats."""
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        "image": 10 * 1024 * 1024,  # 10 MB
        "audio": 50 * 1024 * 1024,  # 50 MB
        "video": 100 * 1024 * 1024,  # 100 MB
        "document": 5 * 1024 * 1024,  # 5 MB
    }
    
    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        "image": [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/bmp",
        ],
        "audio": [
            "audio/mpeg",
            "audio/wav",
            "audio/ogg",
            "audio/flac",
            "audio/mp4",
        ],
        "video": [
            "video/mp4",
            "video/mpeg",
            "video/webm",
            "video/quicktime",
        ],
        "document": [
            "application/pdf",
            "text/plain",
            "application/json",
        ]
    }
    
    # Dangerous file extensions (blocked)
    BLOCKED_EXTENSIONS = {
        ".exe", ".dll", ".bat", ".cmd", ".sh", ".ps1",
        ".vbs", ".js", ".jar", ".app", ".msi", ".com",
        ".scr", ".pif", ".cpl", ".reg"
    }
    
    @staticmethod
    async def validate_file_upload(
        file: UploadFile,
        file_category: str = "image",
        max_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[bytes]]:
        """
        Validate an uploaded file for security.
        
        Args:
            file: The uploaded file
            file_category: Category of file (image, audio, video, document)
            max_size: Optional custom max size in bytes
            
        Returns:
            Tuple of (is_valid, error_message, file_content)
        """
        try:
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Reset file pointer for potential re-reading
            await file.seek(0)
            
            # 1. Check file size
            max_allowed = max_size or FileSecurityValidator.MAX_FILE_SIZES.get(file_category, 5 * 1024 * 1024)
            if file_size > max_allowed:
                return False, f"File size ({file_size} bytes) exceeds maximum allowed ({max_allowed} bytes)", None
            
            # 2. Check filename
            if not file.filename:
                return False, "Filename is required", None
            
            # 3. Check for blocked extensions
            filename_lower = file.filename.lower()
            for blocked_ext in FileSecurityValidator.BLOCKED_EXTENSIONS:
                if filename_lower.endswith(blocked_ext):
                    return False, f"File type {blocked_ext} is not allowed for security reasons", None
            
            # 4. Validate MIME type using magic numbers (not just extension)
            try:
                detected_mime = magic.from_buffer(content, mime=True)
            except Exception as e:
                logger.warning(f"Could not detect MIME type: {str(e)}")
                detected_mime = file.content_type
            
            # 5. Check if MIME type is allowed
            allowed_mimes = FileSecurityValidator.ALLOWED_MIME_TYPES.get(file_category, [])
            if detected_mime not in allowed_mimes:
                return False, f"File type {detected_mime} is not allowed. Allowed types: {', '.join(allowed_mimes)}", None
            
            # 6. Additional checks for specific file types
            if file_category == "image":
                # Check for image bombs (extremely large dimensions)
                if not FileSecurityValidator._validate_image_dimensions(content):
                    return False, "Image dimensions exceed safe limits (potential image bomb)", None
            
            # 7. Sanitize filename
            sanitized_filename = FileSecurityValidator._sanitize_filename(file.filename)
            file.filename = sanitized_filename
            
            # 8. Calculate file hash for deduplication and integrity
            file_hash = hashlib.sha256(content).hexdigest()
            logger.info(f"✅ File validated: {sanitized_filename} (SHA256: {file_hash[:16]}...)")
            
            return True, None, content
            
        except Exception as e:
            logger.error(f"❌ File validation error: {str(e)}")
            return False, f"File validation failed: {str(e)}", None
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal and injection."""
        import re
        
        # Remove path components
        filename = filename.split("/")[-1].split("\\")[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:240] + ('.' + ext if ext else '')
        
        # Ensure filename is not empty
        if not filename or filename == '.':
            filename = "unnamed_file"
        
        return filename
    
    @staticmethod
    def _validate_image_dimensions(content: bytes) -> bool:
        """Validate image dimensions to prevent decompression bombs."""
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(content))
            width, height = image.size
            
            # Maximum 50 megapixels (e.g., 7071x7071)
            MAX_PIXELS = 50_000_000
            
            if width * height > MAX_PIXELS:
                logger.warning(f"⚠️  Image dimensions too large: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Image validation error: {str(e)}")
            return False
    
    @staticmethod
    def validate_and_raise(
        file: UploadFile,
        file_category: str = "image",
        max_size: Optional[int] = None
    ):
        """
        Convenience method that raises HTTPException on validation failure.
        Use in FastAPI endpoints.
        """
        import asyncio
        
        is_valid, error_message, content = asyncio.run(
            FileSecurityValidator.validate_file_upload(file, file_category, max_size)
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {error_message}"
            )
        
        return content


# Convenience instance
file_validator = FileSecurityValidator()
