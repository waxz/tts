import sys
import re
import asyncio
from kokoro import KPipeline
import base_model
import utils

class StreamingEngine(base_model.BaseEngine):
    def __init__(self, name):
        # 1. Initialize configuration variables first
        self.default_voice = "af_heart"
        self.voice_mapping = {
            "alloy": "af_heart",
            "echo": "af_bella",
            "fable": "af_nicole",
            "onyx": "af_aoede",
            "nova": "af_aoede",
            "shimmer": "af_aoede"
        }
        
        # 2. Call super init (which usually calls load_model)
        super().__init__(name)

    def load_model(self):
        try:
            self.tts = KPipeline(lang_code='a')
            # self.text_processor = self.tts.model.text_processor
            self.sample_rate = 24000
            print(f"Model Loaded. Rate: {self.sample_rate}")
        except Exception as e:
            # 3. CRITICAL FIX: Don't sys.exit(1). Raise exception instead.
            print(f"Error initializing model {self.name}: {e}")
            raise RuntimeError(f"Failed to load model {self.name}") from e

    def get_style_safe(self, voice_name: str):
        """
        Safely retrieves a voice style. 
        """
        # 4. Logic optimized: Map -> Try -> Fallback
        clean_name = voice_name.lower().strip()
        target_name = self.voice_mapping.get(clean_name, self.default_voice)
        print(f"Found voice {target_name}")
        return target_name

    def preprocess_text(self, text):
        if not text:
            return []
        return [text]
        
    def generate(self, chunks: str, voice_name: str, speed: float):
        """
        Generates audio.
        Returns: audio_float_array
        """
        
        
        # If supertonic DOES NOT support speed, simple generation:
        generator = self.tts(chunks, voice=voice_name,speed=speed)
        for i, (gs, ps, audio) in enumerate(generator):
            yield audio.numpy()