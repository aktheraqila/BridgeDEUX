import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from jiwer import wer, cer
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


TEST_PARQUET = Path(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
)

MODEL_DIR = Path(
    "experiments/checkpoints/w1_merged"
)

OUTPUT_DIR = Path(
    "experiments/results/w1_hf_controlled"
)

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_BEAMS = 5
TEMPERATURE = 0.0

PROGRESS_INTERVAL = 25


def normalize_text(text: str) -> str:
    text = re.sub(
        r"<[^>]*>",
        " ",
        str(text),
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.category(c).startswith("P")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def load_model():
    print(f"Device: {DEVICE}")
    print(f"Model : {MODEL_DIR}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_DIR,
        language="german",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


@torch.no_grad()
def transcribe(
    processor,
    model,
    audio_path: str,
) -> str:

    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(DEVICE)

    generated_ids = model.generate(
        input_features=input_features,
        language="de",
        task="transcribe",
        num_beams=NUM_BEAMS,
        do_sample=False,
        temperature=TEMPERATURE,
        return_timestamps=False,
    )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


def evaluate():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(TEST_PARQUET)

    print("=" * 70)
    print("W1 HF — CONTROLLED FULL MSLT TEST EVALUATION")
    print("=" * 70)
    print(f"Samples : {len(df)}")
    print(f"Model   : {MODEL_DIR}")
    print(f"Device  : {DEVICE}")
    print(f"Beams   : {NUM_BEAMS}")
    print(f"Temp    : {TEMPERATURE}")
    print()

    processor, model = load_model()

    results = []

    start_time = time.time()

    for index, row in df.iterrows():

        prediction = transcribe(
            processor,
            model,
            row["audio_path"],
        )

        results.append(
            {
                "id": str(row["id"]),
                "reference": row["t1_reference"],
                "prediction": prediction,
            }
        )

        completed = index + 1

        if completed % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = len(df) - completed
            eta_minutes = (
                remaining / rate / 60
                if rate > 0
                else 0
            )

            print(
                f"[{completed}/{len(df)}] "
                f"rate={rate:.2f}/s "
                f"ETA={eta_minutes:.1f} min"
            )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    references = [
        normalize_text(x)
        for x in result_df["reference"]
    ]

    predictions = [
        normalize_text(x)
        for x in result_df["prediction"]
    ]

    corpus_wer = wer(
        references,
        predictions,
    )

    corpus_cer = cer(
        references,
        predictions,
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    print()
    print("=" * 70)
    print("CONTROLLED HF EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Samples       : {len(result_df)}")
    print(f"WER           : {corpus_wer:.4f} ({corpus_wer * 100:.2f}%)")
    print(f"CER           : {corpus_cer:.4f} ({corpus_cer * 100:.2f}%)")
    print(
        f"Exact matches : "
        f"{sum(a == b for a, b in zip(references, predictions))}"
    )
    print(f"Total time    : {elapsed_minutes:.1f} min")
    print(f"Predictions   : {PREDICTIONS_FILE}")
    print("=" * 70)


def main():
    evaluate()


if __name__ == "__main__":
    main()