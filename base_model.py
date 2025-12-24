import asyncio
import utils

class BaseEngine:
    def __init__(self, name):
        self.lock = asyncio.Lock()
        self.name = name
        self.tts = None
        # Initialize with default, subclass should overwrite or load_model should update it
        self.sample_rate = 24000 

        print(f"Init model {self.name}")
        self.load_model()

    def load_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_style_safe(self, voice_name: str):
        raise NotImplementedError("Subclass must implement abstract method")
    
    # FIX: Changed from async to sync because it's run in an executor
    # FIX: Fixed typo 'genetrate' -> 'generate'
    def generate(self, chunks: str, voice_name: str, speed: float):
        """
        Should return (audio_float_array, sample_rate)
        This method is CPU blocking, so it stays synchronous.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    # FIX: Added a default preprocessor in case subclass doesn't have one
    def preprocess_text(self, text: str):
        return text
    async def stream_generator(self, text: str, voice_name: str, speed: float, format: str):
        encoder = None

        if format == "wav":
            yield utils.create_wav_header(self.sample_rate)
        elif format == "mp3":
            encoder = utils.create_mp3_encoder(sample_rate=self.sample_rate)

        # Preprocess text and voice
        try:
            voice_name = self.get_style_safe(voice_name)
        except NotImplementedError:
            pass 

        chunks = self.preprocess_text(text)
        
        
        loop = asyncio.get_event_loop()
        
        for i, chunk in enumerate(chunks):
            async with self.lock:
                # Run synchronous generation in executor
                audio_float = await loop.run_in_executor(
                    None, 
                    self.generate, 
                    chunk,
                    voice_name,
                    speed 
                )
            for audio in audio_float:
                if format == "wav":
                    pcm_bytes = utils.float_to_pcm16(audio)
                    yield pcm_bytes
                
                elif format == "mp3":
                    # This now returns 'bytes', so it is safe
                    mp3_bytes = utils.float_to_mp3(audio, encoder)
                    if len(mp3_bytes) > 0:
                        yield mp3_bytes

        # Flush MP3 encoder to get remaining audio frames
        if format == "mp3" and encoder is not None:
            final_data = encoder.flush()
            if len(final_data) > 0:
                yield bytes(final_data) # <--- CRITICAL FIX: Cast to bytes