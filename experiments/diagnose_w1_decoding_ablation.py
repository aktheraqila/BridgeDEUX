import os
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from jiwer import wer
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


MODEL_DIR = Path(
    "experiments/checkpoints/w1_merged"
)

GGML_MODEL = Path(
    "models/w1-kd-hf/ggml-model.bin"
)

WHISPER_CLI = Path(
    "whisper.cpp/build/bin/whisper-cli.exe"
)

DATASET = Path(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
)

OUTPUT_DIR = Path(
    "experiments/results/w1_decoding_ablation"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "decoding_ablation.csv"
)

SAMPLE_IDS = [
    "0431",
    "1899",
    "0429",
    "0285",
    "1378",
    "1613",
]


def normalize_text(text):
    return str(text).strip().lower()


def calculate_wer(reference, hypothesis):
    return wer(
        [normalize_text(reference)],
        [normalize_text(hypothesis)],
    )


@torch.no_grad()
def run_hf(
    processor,
    model,
    audio_path,
    decoding_name,
    num_beams,
    length_penalty=None,
):
    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    kwargs = {
        "input_features": inputs["input_features"],
        "language": "de",
        "task": "transcribe",
        "do_sample": False,
        "temperature": 0.0,
        "return_timestamps": False,
    }

    if num_beams > 1:
        kwargs["num_beams"] = num_beams
    else:
        kwargs["num_beams"] = 1

    if length_penalty is not None:
        kwargs["length_penalty"] = length_penalty

    generated = model.generate(**kwargs)

    prediction = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


def run_ggml(audio_path):
    env = os.environ.copy()

    msys_bin = r"C:\msys64\ucrt64\bin"

    env["PATH"] = (
        msys_bin
        + os.pathsep
        + env.get("PATH", "")
    )

    command = [
        str(WHISPER_CLI),
        "-m",
        str(GGML_MODEL),
        "-f",
        str(audio_path),
        "-l",
        "de",
        "-nt",
        "-np",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"GGML failed for {audio_path}\n"
            f"Return code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    prediction = result.stdout.strip()

    if not prediction:
        raise RuntimeError(
            f"GGML produced empty output for {audio_path}"
        )

    return prediction


def main():

    print("=" * 70)
    print("W1 — DECODING ABLATION DIAGNOSTIC")
    print("=" * 70)

    print()
    print("This test does NOT modify the W1 model.")
    print("Only six pathological samples are evaluated.")

    df = pd.read_parquet(DATASET)

    df["id"] = (
        df["id"]
        .astype(str)
        .str.zfill(4)
    )

    selected = df[
        df["id"].isin(SAMPLE_IDS)
    ].copy()

    selected = (
        selected
        .set_index("id")
        .reindex(SAMPLE_IDS)
        .reset_index()
    )

    if selected["audio_path"].isna().any():
        missing = selected[
            selected["audio_path"].isna()
        ]["id"].tolist()

        raise RuntimeError(
            f"Missing samples: {missing}"
        )

    print()
    print("Loading W1 HF model...")

    processor = (
        WhisperProcessor.from_pretrained(
            MODEL_DIR,
            language="german",
            task="transcribe",
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(MODEL_DIR)
    )

    model.eval()

    print("HF model loaded.")

    configurations = [
        {
            "name": "HF_beam5_default",
            "num_beams": 5,
            "length_penalty": None,
        },
        {
            "name": "HF_beam5_length_-1",
            "num_beams": 5,
            "length_penalty": -1.0,
        },
        {
            "name": "HF_greedy",
            "num_beams": 1,
            "length_penalty": None,
        },
    ]

    results = []

    for _, row in selected.iterrows():

        sample_id = row["id"]
        audio_path = row["audio_path"]
        reference = str(row["t1_reference"])

        print()
        print("=" * 70)
        print(f"SAMPLE {sample_id}")
        print("=" * 70)

        print()
        print("REFERENCE:")
        print(reference)

        for config in configurations:

            print()
            print("-" * 70)
            print(config["name"])
            print("-" * 70)

            prediction = run_hf(
                processor=processor,
                model=model,
                audio_path=audio_path,
                decoding_name=config["name"],
                num_beams=config["num_beams"],
                length_penalty=config["length_penalty"],
            )

            score = calculate_wer(
                reference,
                prediction,
            )

            print("Prediction:")
            print(prediction)

            print()
            print(
                f"WER: {score:.3f}"
            )

            results.append(
                {
                    "id": sample_id,
                    "reference": reference,
                    "system": config["name"],
                    "prediction": prediction,
                    "wer": score,
                    "returncode": 0,
                }
            )

        print()
        print("-" * 70)
        print("GGML")
        print("-" * 70)

        try:
            ggml_prediction = run_ggml(
                audio_path
            )

            ggml_score = calculate_wer(
                reference,
                ggml_prediction,
            )

            print("Prediction:")
            print(ggml_prediction)

            print()
            print(
                f"WER: {ggml_score:.3f}"
            )

            results.append(
                {
                    "id": sample_id,
                    "reference": reference,
                    "system": "GGML",
                    "prediction": ggml_prediction,
                    "wer": ggml_score,
                    "returncode": 0,
                }
            )

        except RuntimeError as exc:

            print()
            print("GGML ERROR:")
            print(exc)

            results.append(
                {
                    "id": sample_id,
                    "reference": reference,
                    "system": "GGML",
                    "prediction": "",
                    "wer": None,
                    "returncode": -1,
                }
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pivot = results_df.pivot(
        index="id",
        columns="system",
        values="wer",
    )

    print()
    print(
        pivot.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()