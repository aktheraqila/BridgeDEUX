import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForTDT

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
AUDIO = "datasets/raw/mslt/de_en/test/MSLT_Test_DE_0001.T0.de.wav"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForTDT.from_pretrained(
    MODEL_ID,
    dtype="auto",
)
model.eval()

print("Loading audio...")
audio, sample_rate = sf.read(AUDIO)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

print(f"Sample rate: {sample_rate}")
print(f"Samples: {len(audio)}")
print("Running Parakeet HF inference...")

inputs = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

print("\nOutput type:")
print(type(outputs))

print("\nSequences:")
print(outputs.sequences)

print("\nDecoding...")

text = processor.batch_decode(
    outputs.sequences,
    skip_special_tokens=True,
)

print("\nTRANSCRIPTION:")
print(text)

print("\nDONE")