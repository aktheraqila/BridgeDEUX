from pathlib import Path

import pandas as pd
import torch
import soundfile as sf

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "openai/whisper-base"

DEV_PARQUET = Path("datasets/cache/mslt/de_en/dev/mslt_de_en_dev.parquet")
ANALYZED_PARQUET = Path("datasets/cache/mslt/de_en/dev/parakeet_dev_analyzed.parquet")

NUM_SAMPLES = 10


def main():
    print("=" * 60)
    print("WHISPER LoRA SMOKE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load MSLT Dev data
    # ---------------------------------------------------------
    print("\n[1/6] Loading MSLT Dev data...")

    dev_df = pd.read_parquet(DEV_PARQUET)
    analyzed_df = pd.read_parquet(ANALYZED_PARQUET)

    print(f"Dev rows      : {len(dev_df)}")
    print(f"Analyzed rows : {len(analyzed_df)}")

    train_ids = set(
        analyzed_df.loc[
            analyzed_df["subset"].astype(str).str.lower() == "train",
            "id",
        ].astype(str)
    )

    samples = dev_df[
        dev_df["id"].astype(str).isin(train_ids)
    ].head(NUM_SAMPLES).copy()

    if len(samples) == 0:
        raise RuntimeError("No training samples found.")

    print(f"Smoke samples: {len(samples)}")

    # ---------------------------------------------------------
    # 2. Load Whisper
    # ---------------------------------------------------------
    print("\n[2/6] Loading Whisper processor...")

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="german",
        task="transcribe",
    )

    print("Loading Whisper model...")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---------------------------------------------------------
    # 3. Freeze encoder + apply LoRA
    # ---------------------------------------------------------
    print("\n[3/6] Applying LoRA...")

    for param in model.model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "out_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # ---------------------------------------------------------
    # 4. Load actual MSLT audio
    # ---------------------------------------------------------
    print("\n[4/6] Loading audio...")

    sample = samples.iloc[0]

    sample_id = sample["id"]
    audio_path = Path(sample["audio_path"])

    reference = str(sample["clean_transcript"])

    print(f"Sample ID : {sample_id}")
    print(f"Audio     : {audio_path}")
    print(f"Reference : {reference}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist:\n{audio_path}")

    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"Sample rate : {sample_rate}")
    print(f"Duration    : {len(audio) / sample_rate:.2f}s")

    # ---------------------------------------------------------
    # 5. Forward pass
    # ---------------------------------------------------------
    print("\n[5/6] Running forward pass...")

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features

    print(f"Input features: {tuple(input_features.shape)}")

    labels = processor.tokenizer(
        reference,
        return_tensors="pt",
    ).input_ids

    print(f"Labels: {tuple(labels.shape)}")

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            labels=labels,
        )

    print(f"Forward loss: {outputs.loss.item():.6f}")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")

    # ---------------------------------------------------------
    # 6. Backward pass
    # ---------------------------------------------------------
    print("\n[6/6] Running backward pass...")

    model.train()

    outputs = model(
        input_features=input_features,
        labels=labels,
    )

    loss = outputs.loss

    print(f"Backward loss: {loss.item():.6f}")

    loss.backward()

    gradient_parameters = 0
    lora_gradients = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_parameters += 1
            if "lora_" in name:
                lora_gradients += 1
                print(f"LoRA gradient: {name} | norm={param.grad.norm().item():.6f}")

    print()
    print("=" * 60)
    print("SMOKE TEST RESULT")
    print("=" * 60)

    print(f"Parameters with gradients : {gradient_parameters}")
    print(f"LoRA parameters w/ grads  : {lora_gradients}")

    if lora_gradients == 0:
        raise RuntimeError("FAILED: No LoRA parameters received gradients.")

    print("PASS: Whisper + MSLT + LoRA + backward works.")
    print("No training was performed.")


if __name__ == "__main__":
    main()