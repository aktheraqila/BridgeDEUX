from transformers import WhisperForConditionalGeneration

def check_whisper_params():
    print("Loading openai/whisper-base...")
    m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    total_params = sum(p.numel() for p in m.parameters())
    embed_params = m.model.decoder.embed_tokens.weight.numel()
    vocab_size = m.config.vocab_size
    d_model = m.config.d_model
    
    print("=" * 60)
    print(" WHISPER-BASE PARAMETER BREAKDOWN")
    print("=" * 60)
    print(f"Total Parameters     : {total_params / 1e6:.2f} M")
    print(f"Embedding Parameters : {embed_params / 1e6:.2f} M")
    print(f"Embedding Share      : {100 * embed_params / total_params:.2f}%")
    print(f"Vocab Dimensions     : {vocab_size} tokens × {d_model} dim")
    print("=" * 60)

if __name__ == "__main__":
    check_whisper_params()