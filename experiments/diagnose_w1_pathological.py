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
    "experiments/results/w1_pathological"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "pathological_comparison.csv"
)


PATHOLOGICAL_IDS = [
    "0431",
    "1899",
    "0429",
    "0285",
    "1378",
    "1613",
]


def normalize_text(text):
    return str(text).strip().lower()


def run_hf(
    processor,
    model,
    audio_path,
):
    audio, sample_rate = sf.read(audio_path)

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated = model.generate(
            inputs["input_features"],
            language="de",
            task="transcribe",
            num_beams=5,
            do_sample=False,
            temperature=0.0,
            return_timestamps=False,
        )

    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()


def run_ggml(audio_path):
    """
    Use the same whisper.cpp invocation that succeeded
    in the full W1 GGML evaluation.

    The MSYS2 UCRT64 runtime DLL directory is explicitly
    placed first in PATH so Windows does not accidentally
    load the incompatible MinGW runtime.
    """

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

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # whisper.cpp normally prints transcription to stdout.
    # Do not treat an empty output as a valid transcription.
    if result.returncode != 0:
        raise RuntimeError(
            "whisper.cpp failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    if not stdout:
        raise RuntimeError(
            "whisper.cpp returned exit code 0 "
            "but produced empty stdout."
        )

    return stdout, result.returncode


def calculate_wer(
    reference,
    hypothesis,
):
    return wer(
        [normalize_text(reference)],
        [normalize_text(hypothesis)],
    )


def main():
    print("=" * 70)
    print("W1 — PATHOLOGICAL HF vs GGML DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Dataset:", DATASET)
    print("HF model:", MODEL_DIR)
    print("GGML model:", GGML_MODEL)
    print("CLI:", WHISPER_CLI)

    print()
    print("Checking files...")

    required_files = [
        DATASET,
        MODEL_DIR / "config.json",
        GGML_MODEL,
        WHISPER_CLI,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print("OK:", path)

    df = pd.read_parquet(DATASET)

    df["id"] = (
        df["id"]
        .astype(str)
        .str.zfill(4)
    )

    selected = df[
        df["id"].isin(PATHOLOGICAL_IDS)
    ].copy()

    selected = (
        selected
        .set_index("id")
        .reindex(PATHOLOGICAL_IDS)
        .reset_index()
    )

    if selected["audio_path"].isna().any():
        missing = selected[
            selected["audio_path"].isna()
        ]["id"].tolist()

        raise RuntimeError(
            f"Missing sample IDs: {missing}"
        )

    print()
    print("Samples:", len(selected))
    print(
        "IDs:",
        ", ".join(PATHOLOGICAL_IDS),
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

    # Test the actual GGML executable before starting.
    print()
    print("GGML executable:", WHISPER_CLI)
    print("GGML model:", GGML_MODEL)

    results = []

    for _, row in selected.iterrows():

        sample_id = row["id"]
        audio_path = row["audio_path"]
        reference = str(
            row["t1_reference"]
        )

        print()
        print("=" * 70)
        print(f"SAMPLE {sample_id}")
        print("=" * 70)

        print()
        print("Audio:")
        print(audio_path)

        print()
        print("Reference:")
        print(reference)

        # --------------------------------------------------
        # HF
        # --------------------------------------------------

        print()
        print("Running HF...")

        hf_prediction = run_hf(
            processor,
            model,
            audio_path,
        )

        print()
        print("HF:")
        print(hf_prediction)

        hf_wer = calculate_wer(
            reference,
            hf_prediction,
        )

        print()
        print(
            f"HF WER: {hf_wer:.3f}"
        )

        # --------------------------------------------------
        # GGML
        # --------------------------------------------------

        print()
        print("Running GGML...")

        try:
            (
                ggml_prediction,
                return_code,
            ) = run_ggml(audio_path)

            ggml_error = ""

        except RuntimeError as exc:
            ggml_prediction = ""
            return_code = -1
            ggml_error = str(exc)

            print()
            print("GGML ERROR:")
            print(ggml_error)

        print()
        print("GGML:")

        if ggml_prediction:
            print(ggml_prediction)
        else:
            print("<NO VALID OUTPUT>")

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        if ggml_prediction:
            ggml_wer = calculate_wer(
                reference,
                ggml_prediction,
            )

            if hf_wer < ggml_wer:
                winner = "HF"
            elif ggml_wer < hf_wer:
                winner = "GGML"
            else:
                winner = "TIE"
        else:
            ggml_wer = None
            winner = "INVALID GGML"

        print()
        print(
            f"GGML WER: "
            f"{ggml_wer:.3f}"
            if ggml_wer is not None
            else "GGML WER: INVALID"
        )

        print(
            "Winner:",
            winner,
        )

        results.append(
            {
                "id": sample_id,
                "reference": reference,
                "hf_prediction": hf_prediction,
                "ggml_prediction": ggml_prediction,
                "hf_wer": hf_wer,
                "ggml_wer": ggml_wer,
                "ggml_returncode": return_code,
                "ggml_error": ggml_error,
                "winner": winner,
            }
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(
        "ID       HF WER      GGML WER    Result"
    )

    print("-" * 55)

    for result in results:

        hf_value = (
            f"{result['hf_wer']:.3f}"
        )

        if result["ggml_wer"] is None:
            ggml_value = "INVALID"
        else:
            ggml_value = (
                f"{result['ggml_wer']:.3f}"
            )

        print(
            f"{result['id']:<8}"
            f"{hf_value:<12}"
            f"{ggml_value:<12}"
            f"{result['winner']}"
        )

    valid_ggml = [
        r
        for r in results
        if r["ggml_wer"] is not None
    ]

    print()

    print(
        "Valid GGML samples:",
        len(valid_ggml),
        "/",
        len(results),
    )

    if valid_ggml:
        hf_wins = sum(
            r["winner"] == "HF"
            for r in valid_ggml
        )

        ggml_wins = sum(
            r["winner"] == "GGML"
            for r in valid_ggml
        )

        ties = sum(
            r["winner"] == "TIE"
            for r in valid_ggml
        )

        print(
            "HF wins   :",
            hf_wins,
        )

        print(
            "GGML wins :",
            ggml_wins,
        )

        print(
            "Ties      :",
            ties,
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()