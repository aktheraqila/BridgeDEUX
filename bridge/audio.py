# bridge/audio.py
from __future__ import annotations

import io
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

class AudioProcessor:
    """
    Handles memory-isolated audio conversion operations for the BridgeDEUX framework.
    Strictly isolated to raw I/O decoding and standard polyphase resampling.
    """

    @staticmethod
    #def decode_mp3_to_pcm(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    def decode_to_pcm(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """
        Decodes raw MP3 binary data into a mono float32 NumPy array.
        """
        byte_stream = io.BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(byte_stream, dtype='float32')
        
        # If stereo, average the channels to make it mono
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
            
        return audio_array, sample_rate

    @staticmethod
    def resample_to_16k(audio_array: np.ndarray, current_sample_rate: int) -> np.ndarray:
        """
        Resamples a mono audio array down to 16,000 Hz using high-quality 
        polyphase filtering via SciPy to prevent frequency aliasing.
        """
        if current_sample_rate == 16000:
            return audio_array

        target_sample_rate = 16000
        
        common_divisor = gcd(current_sample_rate, target_sample_rate)
        up = target_sample_rate // common_divisor
        down = current_sample_rate // common_divisor
        
        resampled = resample_poly(audio_array, up, down)
        return resampled.astype(np.float32)