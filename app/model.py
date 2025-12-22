# app/model.py
import torch
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

MODEL_PATH = "models/whisper-tiny-onnx"

def load_model():
    print("Loading Whisper ONNX model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        provider="CPUExecutionProvider"
    )
    print("Model loaded successfully.")
    return model

#uvicorn main:app --host 0.0.0.0 --port 8000
