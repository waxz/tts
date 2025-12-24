import numpy as np
import re
import struct
import lameenc

def split_text_into_sentences(text: str, min_chunk_size: int = 150,split_pattern = r'\n+'):
    if not text:
        return []

    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()

    # Split atomic sentences
    raw_parts = re.split(r'([.?!:;]+)(?=\s|$)', text)
    
    atomic_sentences = []
    current_atomic = ""
    
    for part in raw_parts:
        if not part.strip():
            continue
        if re.match(r'^[.?!:;]+$', part):
            current_atomic += part
            if current_atomic.strip():
                atomic_sentences.append(current_atomic.strip())
            current_atomic = ""
        else:
            current_atomic += part
            
    if current_atomic.strip():
        atomic_sentences.append(current_atomic.strip())

    # Batching Logic
    final_chunks = []
    current_buffer = ""
    first_sentence_sent = False

    for sentence in atomic_sentences:
        if not first_sentence_sent:
            final_chunks.append(sentence)
            first_sentence_sent = True
            continue

        if current_buffer:
            current_buffer += " " + sentence
        else:
            current_buffer = sentence

        if len(current_buffer) >= min_chunk_size:
            final_chunks.append(current_buffer)
            current_buffer = ""

    if current_buffer:
        final_chunks.append(current_buffer)

    return final_chunks

def create_wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16):
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
    
    # Clip to prevent distortion
    audio_array = np.clip(audio_array, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_array * 32767).astype(np.int16)
    return audio_int16.tobytes()

def create_mp3_encoder(sample_rate=44100, channels=1, bit_rate=128, quality=5):
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bit_rate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(quality) 
    return encoder


def float_to_mp3(audio_array, encoder):
    """
    Converts float32 audio -> Int16 -> Encoded MP3 bytes.
    """
    # 1. Convert Float to PCM Int16
    audio_array = np.array(audio_array)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.flatten()
    
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # 2. Encode to MP3
    # lameenc returns a bytearray, but FastAPI/Starlette requires strictly 'bytes'
    mp3_data = encoder.encode(audio_int16.tobytes())
    
    return bytes(mp3_data)  # <--- CRITICAL FIX: Convert bytearray to bytes