import torch
import json
from transformers import WhisperForConditionalGeneration

print("Loading merged model...")
model = WhisperForConditionalGeneration.from_pretrained("experiments/checkpoints/w1_merged")

# Extract dims from model config
dims = {
    "n_mels": model.config.num_mel_bins,
    "n_vocab": model.config.vocab_size,
    "n_audio_ctx": model.config.max_source_positions,
    "n_audio_state": model.config.d_model,
    "n_audio_head": model.config.encoder_attention_heads,
    "n_audio_layer": model.config.encoder_layers,
    "n_text_ctx": model.config.max_target_positions,
    "n_text_state": model.config.d_model,
    "n_text_head": model.config.decoder_attention_heads,
    "n_text_layer": model.config.decoder_layers,
}

# Get model state dict
state_dict = model.model.state_dict()

# Create the expected checkpoint format
checkpoint = {
    "dims": dims,
    "model_state_dict": state_dict,
}

# Save
torch.save(checkpoint, "experiments/checkpoints/w1_model_correct.pt")

print("Saved correct format checkpoint")
print(f"Dims: {dims}")