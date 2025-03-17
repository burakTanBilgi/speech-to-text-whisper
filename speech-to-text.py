from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import uuid
import shutil
from typing import List
from pydantic import BaseModel

app = FastAPI(
    title="Speech-to-Text API with Whisper",
    description="Turkish Speech-to-Text API using OpenAI's Whisper model"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Available models
MODELS = ["tiny", "base", "small", "medium", "large"]

# Model cache to avoid reloading
model_cache = {}

class TranscriptionResponse(BaseModel):
    text: str
    filename: str
    language: str
    model: str

def cleanup_file(file_path: str):
    """Remove temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form("tr")
):
    """
    Transcribe an audio file using Whisper.
    
    - **file**: Audio file to transcribe (mp3, wav, etc.)
    - **model_name**: Whisper model to use (default: base)
    - **language**: Language code (default: tr for Turkish)
    """
    # Validate model name
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be one of {MODELS}")
    
    # Generate a unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = f"{TEMP_DIR}/{uuid.uuid4()}{file_ext}"
    
    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add cleanup task
        background_tasks.add_task(cleanup_file, temp_file_path)
        
        # Load model (with caching)
        if model_name not in model_cache:
            print(f"Loading Whisper model: {model_name}")
            model_cache[model_name] = whisper.load_model(model_name)
        
        model = model_cache[model_name]
        
        # Transcribe audio
        print(f"Transcribing file: {temp_file_path} in language: {language}")
        result = model.transcribe(temp_file_path, language=language)
        
        return {
            "text": result["text"],
            "filename": file.filename,
            "language": language,
            "model": model_name
        }
        
    except Exception as e:
        # Ensure cleanup happens even on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

@app.get("/models/")
async def get_available_models():
    """Get a list of available Whisper models"""
    return {"models": MODELS}

@app.get("/health/")
async def health_check():
    """Check if the API is running and which models are loaded"""
    return {
        "status": "healthy",
        "models_loaded": list(model_cache.keys())
    }

@app.get("/")
async def root():
    """API root"""
    return {"message": "Speech-to-Text API with Whisper. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)