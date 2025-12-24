import os
import argparse
import uvicorn
import sys
import secrets
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Literal
import supertonic_model,kokoro_model



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

# -----------------------------------------------------------------------------
# 2. Streaming Engine with Fallback Logic
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 3. API Setup
# -----------------------------------------------------------------------------

engine = {}

class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: str = "alloy" # Default 'alloy'
    format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Check if API Key is set
    if not os.getenv("API_KEY"):
        print("\n!!! WARNING: API_KEY not set. API is open to the public. !!!\n")
    else:
        print(f"\n*** Secure Mode: API Key protection enabled. ***\n")

    MODELS = None
    if not os.getenv("MODELS"):
        print(f"\n!!! WARNING: MODELS not set")
        sys.exit(0)
    else:
        MODELS = os.getenv("MODELS")
    
    print(f"\n!!! WARNING: eval {MODELS}") 
    try:
        MODELS = eval(MODELS)
    except:
        print(f"\n!!! WARNING: eval {MODELS} failed")        
        sys.exit(0)

    print(f"\n*** Load {MODELS}. ***\n")
    for k,v in MODELS.items():
        print(f"Mapping {k}-->{v}")
        if "supertonic" == v:
            engine[k] = supertonic_model.StreamingEngine(f"{k}-->{v}")
        if "kokoro" == v:
            engine[k] = kokoro_model.StreamingEngine(f"{k}-->{v}")
    yield


app = FastAPI(lifespan=lifespan)


# PROTECTED ROUTE
# The Depends(verify_api_key) enforces auth for this specific endpoint
@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def text_to_speech(request: SpeechRequest):
    global engine
    if not engine:
        raise HTTPException(500, "Engine not loaded")

    print(f"request:{request}")
    format = request.format
    model = request.model
    if format not in ["wav", "mp3"]:
        format = "wav"
    if model not in engine.keys():
        print(f"!!!WARNING {model} not found")
        
        content = {
            "ok": False,
            "message": f"!!!WARNING {model} not found"
        }

        content = json.dumps(content)

        return Response(content=content, status_code=404,media_type="application/json")
        
    

    return StreamingResponse(
        engine[model].stream_generator(request.input, request.voice, request.speed, format),
        media_type=f"audio/{format}"
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
