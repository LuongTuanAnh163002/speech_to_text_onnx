# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import load_model
from app.audio import load_audio
import torch
import requests
import logging
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Whisper ONNX ASR API",
              docs_url=None,
            redoc_url=None, 
            openapi_url=None) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model = None
processor = None

class AudioURL(BaseModel):
    url: str

@app.on_event("startup")
async def startup_event():
    """Load the model and processor during application startup."""
    global model, processor
    try:
        model, processor = load_model()
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model during startup.")

@app.post("/transcribe")
async def transcribe(audio_url: AudioURL):
    """Endpoint to transcribe audio files from a URL."""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Download audio from the provided URL
        response = requests.get(audio_url.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download audio from URL.")

        audio_bytes = response.content
        logger.info("Audio file downloaded successfully from URL: %s", audio_url.url)

        audio = load_audio(audio_bytes)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        logger.info("Transcription completed successfully.")
        return {"text": text}
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing audio: {e}")

@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "api_version": app.version}

@app.get("/")
def api_info():
    return {
        "api_name": app.title,
        "version": app.version,
        "description": app.description,
        "endpoints": {
            "/transcribe": "Speech to text transcription",
            "/health": "Health check",
            "/docs": "Swagger UI",
            "/redoc": "ReDoc"
        }
    }

security = HTTPBasic()

def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("USERNAME_AUTHORIZE")
    correct_password = os.getenv("PASSWORD_AUTHORIZE")
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint(username: str = Depends(verify_user)):
    return get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes
    )

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(username: str = Depends(verify_user)):
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Docs"
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html(username: str = Depends(verify_user)):
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - ReDoc"
    )
