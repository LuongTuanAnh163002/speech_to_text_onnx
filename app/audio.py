import librosa
import tempfile
import os

def load_audio(file_bytes, sr=16000):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    )

    try:
        tmp.write(file_bytes)
        tmp.close()
        audio, _ = librosa.load(tmp.name, sr=sr)
        return audio
    finally:
        os.unlink(tmp.name)
