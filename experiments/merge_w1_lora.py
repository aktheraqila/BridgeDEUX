# merge_w1_lora.py
from pathlib import Path
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

MODEL_ID = "openai/whisper-base"
W1_CHECKPOINT = Path("experiments/checkpoints/w1_unfiltered/best")
OUTPUT_DIR = Path("experiments/checkpoints/w1_merged")

print("Loading Whisper-base...")
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, W1_CHECKPOINT)

print("Merging LoRA into base weights...")
merged = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
merged.save_pretrained(OUTPUT_DIR)

print("Done.")