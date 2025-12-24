import sys
import re
import asyncio
from supertonic import TTS
import base_model

class StreamingEngine(base_model.BaseEngine):
    def __init__(self, name):
        # 1. Initialize configuration variables first
        self.default_voice = "F1"
        self.voice_mapping = {
            "alloy": "F1",
            "echo": "M1",
            "fable": "M2",
            "onyx": "M3",
            "nova": "F2",
            "shimmer": "F3"
        }
        
        # 2. Call super init (which usually calls load_model)
        super().__init__(name)

    def load_model(self):
        try:
            self.tts = TTS(auto_download=True)
            self.text_processor = self.tts.model.text_processor
            self.sample_rate = self.tts.sample_rate
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

        try:
            # Try specific voice
            return self.tts.get_voice_style(voice_name=target_name)
        except Exception:
            print(f"WARNING: Voice '{voice_name}' (mapped to '{target_name}') not found. Using '{self.default_voice}'.")
            
            # Fallback to default
            try:
                return self.tts.get_voice_style(voice_name=self.default_voice)
            except Exception as e:
                # If default fails, we are in trouble
                print(f"CRITICAL: Default voice '{self.default_voice}' also failed.")
                raise e

    def preprocess_text(self, text):
        if not text:
            return ""

        is_valid, unsupported = self.text_processor.validate_text(text)

        if not is_valid:
            print(f"   ⚠️  Contains {len(unsupported)} unsupported character(s): {unsupported[:5]}")
            # Escape characters safe for regex usage
            pattern = f"[{re.escape(''.join(unsupported))}]"
            preprocessed = re.sub(pattern, "", text)
            
            if preprocessed != text:
                print(f"   After preprocessing: {preprocessed[:50]}...")
                text = preprocessed
        else:
            # Optional: Comment this out in production to reduce log spam
            print("   ✓ All characters supported")
            
        return text

    def generate(self, chunks: str, voice_name: str, speed: float):
        """
        Generates audio.
        Returns: audio_float_array
        """
        # 5. Handle Speed (if supported by supertonic, otherwise ignore or warn)
        # Assuming supertonic.synthesize supports a speed or speed_ratio argument:
        # audio = self.tts.synthesize(chunks, voice_name, speed=speed)
        
        # If supertonic DOES NOT support speed, simple generation:
        audio,_ = self.tts.synthesize(chunks, voice_name)
        yield audio