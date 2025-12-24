curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"  -H "Authorization: Bearer yourapi"  -d '{
    "model": "tts-2",
    "input": "Supertonic is a lightning-fast, on-device text-to-speech system designed for extreme performance with minimal computational overhead. Powered by ONNX Runtime, it runs entirely on your device—no cloud, no API calls, no privacy concerns.",
    "voice": "alloy",
    "format": "wav"
  }'   --output ./v2.wav

curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"  -H "Authorization: Bearer yourapi"  -d '{
    "model": "tts-2",
    "input": "Supertonic is a lightning-fast, on-device text-to-speech system designed for extreme performance with minimal computational overhead. Powered by ONNX Runtime, it runs entirely on your device—no cloud, no API calls, no privacy concerns.",
    "voice": "alloy",
    "format": "mp3"
  }'   --output ./v2.mp3

  curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"  -H "Authorization: Bearer yourapi"  -d '{
    "model": "tts-1",
    "input": "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.",
    "voice": "alloy",
    "format": "wav"
  }'   --output ./v1.wav

curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"  -H "Authorization: Bearer yourapi"  -d '{
    "model": "tts-1",
    "input": "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.",
    "voice": "alloy",
    "format": "mp3"
  }'   --output ./v1.mp3