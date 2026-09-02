from pathlib import Path
import re
import unicodedata

import pandas as pd
import soundfile as sf
import torch
from jiwer import wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_ID = "openai/whisper-base"

TEST_PARQUET = Path(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
)

NUM_SAMPLES = 20


def normalize_mslt_t1(text: str) -> str:
    text = str(text)

    # Remove MSLT annotation tags.
    text = re.sub(r"<[^>]*>", " ", text)

    # Remove punctuation.
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )

    # Normalize whitespace and case.
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


@torch.no_grad()
def transcribe_with_mask(processor, model, audio_path):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features

    attention_mask = torch.ones_like(input_features[:, :, 0])

    generated = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        language="german",
        task="transcribe",
        max_length=model.config.max_target_positions,
    )

    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def transcribe_without_mask(processor, model, audio_path):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features

    generated = model.generate(
        input_features=input_features,
        language="german",
        task="transcribe",
        max_length=model.config.max_target_positions,
    )

    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    print("=" * 70)
    print("W0 DIAGNOSTIC: With vs Without Attention Mask")
    print("=" * 70)

    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="german",
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()

    print("Loading test data...")
    test_df = pd.read_parquet(TEST_PARQUET)

    print(f"Testing on first {NUM_SAMPLES} samples\n")

    rows = []

    for idx, (_, sample) in enumerate(
        test_df.head(NUM_SAMPLES).iterrows(),
        start=1,
    ):
        sample_id = str(sample["id"])
        audio_path = Path(sample["audio_path"])
        reference = str(sample["t1_reference"])

        if not audio_path.exists():
            print(f"[{idx}] {sample_id}: AUDIO MISSING")
            continue

        hyp_with = transcribe_with_mask(processor, model, audio_path)
        hyp_without = transcribe_without_mask(processor, model, audio_path)

        ref_norm = normalize_mslt_t1(reference)
        with_norm = normalize_mslt_t1(hyp_with)
        without_norm = normalize_mslt_t1(hyp_without)

        wer_with = wer(ref_norm, with_norm) if ref_norm and with_norm else 1.0
        wer_without = wer(ref_norm, without_norm) if ref_norm and without_norm else 1.0

        rows.append({
            "id": sample_id,
            "reference": reference,
            "hyp_with_mask": hyp_with,
            "hyp_without_mask": hyp_without,
            "wer_with_mask": wer_with,
            "wer_without_mask": wer_without,
        })

        print(f"[{idx}/{NUM_SAMPLES}] {sample_id}")
        print(f"  Ref      : {reference[:80]}")
        print(f"  With mask: {hyp_with[:80]}")
        print(f"  No mask  : {hyp_without[:80]}")
        print(f"  WER with mask: {wer_with:.4f}")
        print(f"  WER no mask  : {wer_without:.4f}")
        print()

    # Summary
    result_df = pd.DataFrame(rows)

    if len(result_df) > 0:
        avg_wer_with = result_df["wer_with_mask"].mean()
        avg_wer_without = result_df["wer_without_mask"].mean()

        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Samples tested        : {len(result_df)}")
        print(f"Average WER with mask : {avg_wer_with:.4f}")
        print(f"Average WER no mask   : {avg_wer_without:.4f}")
        print(f"Difference            : {avg_wer_without - avg_wer_with:+.4f}")

        # How many samples differ?
        diff_count = (
            result_df["hyp_with_mask"] != result_df["hyp_without_mask"]
        ).sum()
        print(f"Samples with different outputs: {diff_count}/{len(result_df)}")

        # Save
        output_path = Path("experiments/results/w0_mask_diagnostic.csv")
        result_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()