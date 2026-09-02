from pathlib import Path
import random

import pandas as pd
import soundfile as sf
import torch
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "openai/whisper-base"

DEV_PARQUET = Path(
    "datasets/cache/mslt/de_en/dev/mslt_de_en_dev.parquet"
)

ANALYZED_PARQUET = Path(
    "datasets/cache/mslt/de_en/dev/parakeet_dev_analyzed.parquet"
)

FILTERED_IDS_FILE = Path(
    "datasets/manifests/mslt_dev_parakeet_filtered_ids.txt"
)

OUTPUT_DIR = Path("experiments/checkpoints/w2_filtered")

EPOCHS = 2
LEARNING_RATE = 5e-5
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 4


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# LoRA
# ============================================================

def build_model():
    print("Loading Whisper...")

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Freeze original encoder weights.
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "out_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model


# ============================================================
# Audio + label preparation
# ============================================================

def prepare_sample(processor, row):
    audio_path = Path(row["audio_path"])

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file does not exist: {audio_path}"
        )

    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features

    # W2 target = Parakeet's FILTERED transcription.
    parakeet_text = str(row["parakeet_transcript"]).strip()

    if not parakeet_text:
        return None

    labels = processor.tokenizer(
        parakeet_text,
        return_tensors="pt",
    ).input_ids

    return input_features, labels


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, processor, df):
    model.eval()

    total_loss = 0.0
    count = 0

    for _, row in df.iterrows():
        prepared = prepare_sample(processor, row)

        if prepared is None:
            continue

        input_features, labels = prepared

        outputs = model(
            input_features=input_features,
            labels=labels,
        )

        loss = outputs.loss.item()

        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(
                f"Non-finite validation loss for sample {row['id']}"
            )

        total_loss += loss
        count += 1

    model.train()

    return total_loss / count if count else float("inf")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("W2 — WHISPER LoRA / FILTERED PARAKEET")
    print("=" * 70)

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\n[1/5] Loading MSLT Dev data...")

    dev_df = pd.read_parquet(DEV_PARQUET)
    analyzed_df = pd.read_parquet(ANALYZED_PARQUET)

    # Merge parakeet_transcript into dev_df
    if "parakeet_transcript" not in dev_df.columns:
        dev_df = dev_df.merge(
            analyzed_df[["id", "parakeet_transcript", "subset"]],
            on="id",
            how="left",
        )

    # Load filtered IDs
    print(f"\nLoading filtered IDs from {FILTERED_IDS_FILE}...")

    with open(FILTERED_IDS_FILE, "r", encoding="utf-8") as f:
        filtered_ids = {
            line.strip()
            for line in f
            if line.strip()
        }

    print(f"Filtered IDs: {len(filtered_ids)}")

    # Get train and validation IDs from analyzed_df
    train_ids = set(
        analyzed_df.loc[
            analyzed_df["subset"].astype(str).str.lower() == "train",
            "id",
        ].astype(str)
    )

    val_ids = set(
        analyzed_df.loc[
            analyzed_df["subset"].astype(str).str.lower() == "validation",
            "id",
        ].astype(str)
    )

    # Intersect train_ids and val_ids with filtered_ids separately
    filtered_train_ids = train_ids.intersection(filtered_ids)
    filtered_val_ids = val_ids.intersection(filtered_ids)

    # Build train and validation DataFrames
    train_df = dev_df[
        dev_df["id"].astype(str).isin(filtered_train_ids)
    ].copy()

    val_df = dev_df[
        dev_df["id"].astype(str).isin(filtered_val_ids)
    ].copy()

    print(f"Dev samples      : {len(dev_df)}")
    print(f"Train (filtered) : {len(train_df)}")
    print(f"Validation (filt): {len(val_df)}")

    required_columns = {
        "id",
        "audio_path",
        "parakeet_transcript",
    }

    missing = required_columns - set(train_df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # 2. Processor + model
    # --------------------------------------------------------

    print("\n[2/5] Loading processor and model...")

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="german",
        task="transcribe",
    )

    model = build_model()

    # --------------------------------------------------------
    # 3. Optimizer
    # --------------------------------------------------------

    print("\n[3/5] Creating optimizer...")

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------

    print("\n[4/5] Starting training...")

    best_val_loss = float("inf")

    global_step = 0

    for epoch in range(1, EPOCHS + 1):

        print()
        print("-" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 70)

        model.train()

        random_indices = list(range(len(train_df)))
        random.shuffle(random_indices)

        running_loss = 0.0
        successful_steps = 0

        optimizer.zero_grad(set_to_none=True)

        for position, index in enumerate(random_indices, start=1):

            row = train_df.iloc[index]

            try:
                prepared = prepare_sample(
                    processor,
                    row,
                )

                if prepared is None:
                    print(
                        f"[{position}/{len(train_df)}] "
                        f"{row['id']} | EMPTY → skipped"
                    )
                    continue

                input_features, labels = prepared

                outputs = model(
                    input_features=input_features,
                    labels=labels,
                )

                loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss for sample {row['id']}"
                    )

                scaled_loss = (
                    loss / GRADIENT_ACCUMULATION_STEPS
                )

                scaled_loss.backward()

                running_loss += loss.item()
                successful_steps += 1

                if (
                    successful_steps
                    % GRADIENT_ACCUMULATION_STEPS
                    == 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=1.0,
                    )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                if position % 50 == 0:
                    average_loss = (
                        running_loss / successful_steps
                        if successful_steps
                        else 0.0
                    )

                    print(
                        f"[{position}/{len(train_df)}] "
                        f"step={global_step} "
                        f"loss={loss.item():.4f} "
                        f"avg={average_loss:.4f}"
                    )

            except Exception as exc:
                raise RuntimeError(
                    f"Training failed on sample "
                    f"{row['id']}: {exc}"
                ) from exc

        # Flush remaining accumulated gradients.
        if successful_steps % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        train_loss = (
            running_loss / successful_steps
            if successful_steps
            else float("inf")
        )

        print()
        print(f"Training loss   : {train_loss:.6f}")
        print("Running validation...")

        val_loss = validate(
            model,
            processor,
            val_df,
        )

        print(f"Validation loss : {val_loss:.6f}")

        # ----------------------------------------------------
        # Save epoch checkpoint
        # ----------------------------------------------------

        epoch_dir = OUTPUT_DIR / f"epoch-{epoch}"

        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        print(f"Checkpoint saved: {epoch_dir}")

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_dir = OUTPUT_DIR / "best"

            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            print(
                f"BEST checkpoint updated "
                f"(validation loss={best_val_loss:.6f})"
            )

    # --------------------------------------------------------
    # 5. Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("W2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss : {best_val_loss:.6f}")
    print(f"Best checkpoint      : {OUTPUT_DIR / 'best'}")
    print(f"Global steps         : {global_step}")
    print("=" * 70)


if __name__ == "__main__":
    main()