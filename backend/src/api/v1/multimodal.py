from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import shutil
import os
from utils.file_handler import MultiModalFileHandler
from models.inference_client import inference_client

router = APIRouter()
file_handler = MultiModalFileHandler()

@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model_key: str = Form("mistral-7b")
):
    """
    Analyze an uploaded file (image, doc, audio, video)
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process file
        result = await file_handler.process_file(temp_path)
        
        # If prompt provided, generate AI response
        if prompt:
            ai_response = await inference_client.multimodal_completion(
                model_key=model_key,
                prompt=prompt,
                file_context=result
            )
            result["ai_analysis"] = ai_response
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
