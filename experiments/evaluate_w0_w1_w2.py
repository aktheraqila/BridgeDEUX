from pathlib import Path
import gc
import re
import unicodedata

import pandas as pd
import soundfile as sf
import torch
from jiwer import wer, cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from benchmarks.checkpoint_manager import CheckpointManager


MODEL_ID = "openai/whisper-base"

TEST_PARQUET = Path(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
)

W1_CHECKPOINT = Path(
    "experiments/checkpoints/w1_unfiltered/best"
)

W2_CHECKPOINT = Path(
    "experiments/checkpoints/w2_filtered/best"
)

OUTPUT_DIR = Path(
    "experiments/results/w0_w1_w2_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = "cpu"

CHECKPOINT_INTERVAL = 25


# ============================================================
# EXACT MSLT T1 NORMALIZATION
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(variant: str):
    print()
    print("=" * 70)
    print(f"Loading {variant}")
    print("=" * 70)

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="german",
        task="transcribe",
    )

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID
    )

    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []

    if variant == "W0":
        model = base_model

    elif variant == "W1":
        model = PeftModel.from_pretrained(
            base_model,
            W1_CHECKPOINT,
        )

    elif variant == "W2":
        model = PeftModel.from_pretrained(
            base_model,
            W2_CHECKPOINT,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# TRANSCRIBE ONE SAMPLE
# ============================================================

@torch.no_grad()
def transcribe(
    processor,
    model,
    audio_path: Path,
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
        language="german",
        task="transcribe",
        max_length=model.config.max_target_positions,
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()


# ============================================================
# EVALUATE ONE MODEL WITH CHECKPOINT MANAGER
# ============================================================

def evaluate_model(
    variant: str,
    test_df: pd.DataFrame,
):

    manager = CheckpointManager(
        model_identifier=f"kd_eval_{variant.lower()}",
        checkpoint_interval=CHECKPOINT_INTERVAL,
        output_dir=OUTPUT_DIR,
    )

    completed_ids = manager.load_completed_samples()

    if completed_ids:
        print(
            f"Resuming {variant}: "
            f"{len(completed_ids)} samples already completed"
        )

    processor, model = load_model(variant)

    total = len(test_df)
    session_processed = 0

    print()
    print(f"Evaluating {variant}: {total} samples")
    print()

    for position, (_, sample) in enumerate(
        test_df.iterrows(),
        start=1,
    ):

        sample_id = str(sample["id"])

        if sample_id in completed_ids:
            continue

        audio_path = Path(sample["audio_path"])

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing audio for {sample_id}: {audio_path}"
            )

        reference = str(sample["t1_reference"])

        try:
            hypothesis = transcribe(
                processor,
                model,
                audio_path,
            )

            reference_norm = normalize_mslt_t1(
                reference
            )

            hypothesis_norm = normalize_mslt_t1(
                hypothesis
            )

            if not reference_norm and not hypothesis_norm:
                sample_wer = 0.0
                sample_cer = 0.0

            elif not reference_norm or not hypothesis_norm:
                sample_wer = 1.0
                sample_cer = 1.0

            else:
                sample_wer = wer(
                    reference_norm,
                    hypothesis_norm,
                )

                sample_cer = cer(
                    reference_norm,
                    hypothesis_norm,
                )

            record = {
                "sample_id": sample_id,
                "model_name": variant,
                "t1_reference": reference,
                "hypothesis": hypothesis,
                "wer": sample_wer,
                "cer": sample_cer,
            }

            manager.save(record)

            completed_ids.add(sample_id)
            session_processed += 1

            if session_processed % 25 == 0:
                print(
                    f"[{session_processed}/{total}] "
                    f"last_wer={sample_wer:.4f}"
                )

        except Exception as exc:
            raise RuntimeError(
                f"{variant} failed on sample "
                f"{sample_id}: {exc}"
            ) from exc

    manager.flush()
    manager.finalize()

    parquet_path = (
        OUTPUT_DIR
        / f"kd_eval_{variant.lower()}"
        / f"kd_eval_{variant.lower()}_results.parquet"
    )

    result_df = pd.read_parquet(parquet_path)

    overall_wer = wer(
        result_df["t1_reference"]
        .map(normalize_mslt_t1)
        .tolist(),
        result_df["hypothesis"]
        .map(normalize_mslt_t1)
        .tolist(),
    )

    overall_cer = cer(
        result_df["t1_reference"]
        .map(normalize_mslt_t1)
        .tolist(),
        result_df["hypothesis"]
        .map(normalize_mslt_t1)
        .tolist(),
    )

    exact = (
        result_df["wer"] == 0.0
    ).sum()

    empty = (
        result_df["hypothesis"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print()
    print("-" * 70)
    print(f"{variant} RESULTS")
    print("-" * 70)
    print(f"Samples       : {len(result_df)}")
    print(f"WER           : {overall_wer:.4f}")
    print(f"CER           : {overall_cer:.4f}")
    print(f"WER = 0       : {exact}")
    print(f"Empty outputs : {empty}")
    print(f"Artifacts     : {parquet_path.parent}")

    del model
    del processor
    del manager

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": variant,
        "samples": len(result_df),
        "wer": overall_wer,
        "cer": overall_cer,
        "wer_zero": int(exact),
        "empty_outputs": int(empty),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("W0 / W1 / W2 — MSLT TEST EVALUATION")
    print("=" * 70)

    print("\nLoading MSLT Test cache...")

    test_df = pd.read_parquet(TEST_PARQUET)

    print(f"Test samples: {len(test_df)}")
    print(
        "Columns:",
        test_df.columns.tolist(),
    )

    required = {
        "id",
        "audio_path",
        "t1_reference",
    }

    missing = required - set(test_df.columns)

    if missing:
        raise RuntimeError(
            f"Missing test columns: {sorted(missing)}"
        )

    results = []

    # --------------------------------------------------------
    # W0
    # --------------------------------------------------------

    results.append(
        evaluate_model(
            "W0",
            test_df,
        )
    )

    # --------------------------------------------------------
    # W1
    # --------------------------------------------------------

    results.append(
        evaluate_model(
            "W1",
            test_df,
        )
    )

    # --------------------------------------------------------
    # W2
    # --------------------------------------------------------

    results.append(
        evaluate_model(
            "W2",
            test_df,
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(results)

    summary_path = OUTPUT_DIR / "summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(
        summary[
            [
                "model",
                "samples",
                "wer",
                "cer",
                "wer_zero",
                "empty_outputs",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()