# tts-proxy
A simple openai api style tts server based on supertonic.

- https://huggingface.co/spaces/Supertone/supertonic
- https://github.com/supertone-inc/supertonic/tree/main/py


## install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv -p 3.10
source .venv/bin/activate
uv pip install -r ./requirements.txt
```

## run server

```bash
python server.py
```

## run client

```bash
curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"   -d '{
    "model": "tts-1",
    "input": "Hello, this is Supertonic running locally!",
    "voice": "F1"
  }'   --output ./test.wav
```