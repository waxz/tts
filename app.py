import os
import uvicorn
import sys
import secrets
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Import your model engines
import supertonic_model
import kokoro_model

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Map config names to Model Classes
MODEL_FACTORIES = {
    "supertonic": supertonic_model.StreamingEngine,
    "kokoro": kokoro_model.StreamingEngine
}

# Global storage for loaded engines
engines: Dict[str, object] = {}

# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    server_key = os.getenv("API_KEY")
    
    if not server_key:
        # Warning already logged in lifespan, safe to pass here for dev mode
        return True

    client_key = credentials.credentials
    if not secrets.compare_digest(server_key, client_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: str = "alloy"
    format: Optional[str] = "mp3" # OpenAI defaults to mp3 usually
    speed: Optional[float] = 1.0

# -----------------------------------------------------------------------------
# Lifecycle (Startup/Shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines

    # 1. API Key Check
    if not os.getenv("API_KEY"):
        logger.warning("API_KEY not set. API is open to the public.")
    else:
        logger.info("Secure Mode: API Key protection enabled.")

    # 2. Load Models Configuration
    models_env = os.getenv("MODELS")
    if not models_env:
        logger.error("MODELS environment variable not set. Exiting.")
        sys.exit(1)

    try:
        # SECURITY FIX: Use json.loads instead of eval
        models_config = json.loads(models_env)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MODELS JSON: {e}")
        sys.exit(1)

    # 3. Initialize Engines
    logger.info(f"Loading models configuration: {models_config}")
    
    for model_id, backend_type in models_config.items():
        if backend_type not in MODEL_FACTORIES:
            logger.error(f"Unknown backend type '{backend_type}' for model '{model_id}'")
            continue

        try:
            logger.info(f"Initializing {model_id} -> {backend_type}...")
            engine_class = MODEL_FACTORIES[backend_type]
            engines[model_id] = engine_class(f"{model_id}-->{backend_type}")
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            # Optional: sys.exit(1) if you want strict startup failure

    if not engines:
        logger.error("No engines loaded successfully. Exiting.")
        sys.exit(1)

    yield
    
    # Cleanup (if needed)
    engines.clear()

app = FastAPI(lifespan=lifespan, title="Streaming TTS API")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def text_to_speech(request: SpeechRequest):
    global engines
    
    if not engines:
        raise HTTPException(status_code=500, detail="No TTS engines loaded")

    # Validate Model
    if request.model not in engines:
        valid_models = list(engines.keys())
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Model '{request.model}' not found. Available: {valid_models}",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    # Validate Format
    audio_format = request.format if request.format else "mp3"
    if audio_format not in ["wav", "mp3"]:
        audio_format = "wav" # Fallback

    logger.info(f"Generating: model={request.model} voice={request.voice} fmt={audio_format} len={len(request.input)}")

    try:
        generator = engines[request.model].stream_generator(
            request.input, 
            request.voice, 
            request.speed, 
            audio_format
        )
        
        return StreamingResponse(
            generator,
            media_type=f"audio/{audio_format}"
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """
    Returns the list of currently loaded models dynamically.
    """
    model_list = []
    for model_id, engine_inst in engines.items():
        # Try to get inner name if available, else use backend name
        owned_by = getattr(engine_inst, "name", "system")
        model_list.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": owned_by
        })
    
    return {"object": "list", "data": model_list}

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # It's better to run uvicorn from CLI, but this supports python app.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)