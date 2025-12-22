# Whisper ONNX ASR API

This project provides an API for automatic speech recognition (ASR) using the Whisper ONNX model. The API is built with FastAPI and supports audio transcription.

## Features
- Transcribe audio files to text using the `/transcribe` endpoint.
- Utilizes the Whisper ONNX model for efficient and accurate transcription.

## Requirements
- Python 3.9+
- Docker (optional, for containerized deployment)

## Installation

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd asr-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Access the API documentation at `http://127.0.0.1:8000/docs`.

### Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t whisper-asr-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 whisper-asr-api
   ```

## Usage

### Transcription Endpoint
- Endpoint: `POST /transcribe`
- Description: Upload an audio file to transcribe it into text.
- Example using `curl`:
  ```bash
  curl -X POST "http://127.0.0.1:8000/transcribe" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@your-audio-file.mp3"
  ```

## Testing
1. Ensure the application is running locally or in Docker.
2. Run the test script:
   ```bash
   python test/test_api.py
   ```

## Project Structure
- `main.py`: Entry point for the FastAPI application.
- `app/`: Contains the audio processing and model loading logic.
- `models/`: Stores the Whisper ONNX model files.
- `test/`: Contains test scripts for the API.

## License
This project is licensed under the MIT License.