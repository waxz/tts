import os
import io
import time
import re
import asyncio
import numpy as np
import argparse
import uvicorn
import sys
import struct
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal
from supertonic import TTS




# -----------------------------------------------------------------------------
# 1. Authentication Logic
# -----------------------------------------------------------------------------

# Standard Bearer token scheme (used by OpenAI clients)
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verifies that the Bearer token sent by the client matches the API_KEY env var.
    """
    server_key = os.getenv("API_KEY")
    
    # If no key is set on the server, we can either:
    # A) Block everything (Safe default)
    # B) Allow everything (Dev mode)
    # Let's Allow everything but print a warning if no key is configured.
    if not server_key:
        # print("WARNING: No API_KEY set. Allowing unauthenticated request.")
        return True

    client_key = credentials.credentials
    
    # Secure string comparison to prevent timing attacks
    if not secrets.compare_digest(server_key, client_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# -----------------------------------------------------------------------------
# 2. Text & Audio Utilities
# -----------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\*.*?\*', '', text) # Remove actions
    # Remove Emojis/Symbols (Fixed SyntaxError from before)
    text = re.sub(r"[^\w\s,.:;?!'\"\-\u00C0-\u00FF]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_sentences(text: str):
    parts = re.split(r'([.?!]+)', text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.search(r'[.?!]', part):
            if current.strip(): sentences.append(current.strip())
            current = ""
    if current.strip(): sentences.append(current.strip())
    return sentences

# -----------------------------------------------------------------------------
# 1. Utility Functions
# -----------------------------------------------------------------------------

def split_text_into_sentences2(text: str):
    """
    Splits text into chunks (sentences) for streaming.
    """
    parts = re.split(r'([.?!]+)', text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.search(r'[.?!]', part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences

def create_wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16):
    """
    Generates a generic WAV header with "unknown" file size (0xFFFFFFFF)
    so browsers/clients treat it as a stream.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = b'RIFF'
    header += struct.pack('<I', 0xFFFFFFFF) 
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16) 
    header += struct.pack('<H', 1) 
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', 0xFFFFFFFF)
    
    return header

def float_to_pcm16(audio_array):
    """Converts float32 audio to int16 bytes."""
    audio_array = np.array(audio_array)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.flatten()
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    return audio_int16.tobytes()

# -----------------------------------------------------------------------------
# 2. Streaming Engine with Fallback Logic
# -----------------------------------------------------------------------------

class StreamingEngine:
    def __init__(self):
        self.model = None
        self.sample_rate = 441000
        self.lock = asyncio.Lock()
        
        # Default fallback voice
        self.default_voice = "F1" 
        
        # Mapping OpenAI voice names to Supertonic IDs
        self.voice_mapping = {
            "alloy": "F1",
            "echo": "M1",
            "fable": "M2",
            "onyx": "M3",
            "nova": "F2",
            "shimmer": "F3"
        }

        print(f"Loading Supertonic model...")
        try:
            self.model = TTS(auto_download=True)
            self.sample_rate = self.model.sample_rate
            print(f"Model Loaded. Rate: {self.sample_rate}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            sys.exit(1)

    def get_style_safe(self, voice_name: str):
        """
        Safely retrieves a voice style. 
        1. Checks mapping (alloy -> F1).
        2. Tries to load.
        3. If fails, returns default (F1).
        """
        # 1. Normalize and Map
        clean_name = voice_name.lower().strip()
        target_name = self.voice_mapping.get(clean_name, voice_name) # map or keep original

        # 2. Try to get style
        try:
            # Note: We rely on supertonic throwing an error if name is invalid
            style = self.model.get_voice_style(voice_name=target_name)
            return style, target_name
        except Exception:
            # 3. Fallback
            print(f"WARNING: Voice '{voice_name}' (mapped to '{target_name}') not found. Using '{self.default_voice}'.")
            try:
                style = self.model.get_voice_style(voice_name=self.default_voice)
                return style, self.default_voice
            except Exception as e:
                print(f"CRITICAL: Default voice '{self.default_voice}' also failed.")
                raise e

    async def stream_generator(self, text: str, voice_name: str, speed: float):
        # 1. Resolve Voice Style ONCE before the loop
        # We do this here so we don't re-calculate embedding for every sentence
        try:
            style, resolved_name = self.get_style_safe(voice_name)
        except Exception as e:
            print(f"Error resolving voice: {e}")
            return

        yield create_wav_header(self.sample_rate)

        chunks = split_text_into_sentences(text)
        print(f"Streaming '{text[:20]}...' using voice: {resolved_name}")
        
        loop = asyncio.get_event_loop()
        
        for i, chunk in enumerate(chunks):
            # async with self.lock guarantees only one heavy TTS task runs globally
            async with self.lock:
                audio_float, _ = await loop.run_in_executor(
                    None, 
                    self.model.synthesize,
                    chunk,
                    style
                    # speed # Add speed here if your supertonic version supports it
                )

            pcm_bytes = float_to_pcm16(audio_float)
            yield pcm_bytes

# -----------------------------------------------------------------------------
# 3. API Setup
# -----------------------------------------------------------------------------

engine = None

class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: str = "F1" # Defaults to F1, but handles 'alloy' etc via mapping
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Check if API Key is set
    if not os.getenv("API_KEY"):
        print("\n!!! WARNING: API_KEY not set. API is open to the public. !!!\n")
    else:
        print(f"\n*** Secure Mode: API Key protection enabled. ***\n")
        
    engine = StreamingEngine()
    yield


app = FastAPI(lifespan=lifespan)


# PROTECTED ROUTE
# The Depends(verify_api_key) enforces auth for this specific endpoint
@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def text_to_speech(request: SpeechRequest):
    global engine
    if not engine:
        raise HTTPException(500, "Engine not loaded")

    return StreamingResponse(
        engine.stream_generator(request.input, request.voice, request.speed),
        media_type="audio/wav"
    )

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "tts-1", "owned_by": "supertonic"}]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
