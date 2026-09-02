import pandas as pd
import soundfile as sf
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration


DATASET = "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
MODEL_DIR = "experiments/checkpoints/w1_merged"
NUM_SAMPLES = 10


def load_model():
    processor = WhisperProcessor.from_pretrained(
        MODEL_DIR,
        language="german",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return processor, model


def transcribe(processor, model, audio, sampling_rate):
    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
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


def main():
    print("=" * 70)
    print("W1 HF — CONTROLLED DECODING TEST")
    print("=" * 70)

    df = pd.read_parquet(DATASET).head(NUM_SAMPLES)

    processor, model = load_model()

    for _, row in df.iterrows():
        audio, sampling_rate = sf.read(row["audio_path"])

        prediction = transcribe(
            processor,
            model,
            audio,
            sampling_rate,
        )

        print()
        print(f"[{row['id']}]")
        print("REF:", row["t1_reference"])
        print("HF :", prediction)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
