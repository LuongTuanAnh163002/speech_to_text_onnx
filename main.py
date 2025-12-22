# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import load_model
from app.audio import load_audio
import torch

# Initialize FastAPI app
app = FastAPI(title="Whisper ONNX ASR API")

# Global variables for model and processor
model = None
processor = None

@app.on_event("startup")
async def startup_event():
    """Load the model and processor during application startup."""
    global model, processor
    try:
        model = load_model()
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model during startup.")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Endpoint to transcribe audio files."""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        audio_bytes = await file.read()
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

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {e}")
