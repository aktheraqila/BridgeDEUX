import subprocess
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

df = pd.read_parquet(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
).head(10)

model_dir = "experiments/checkpoints/w1_merged"
ggml_model = "models/w1-kd-hf/ggml-model.bin"
cli = "whisper.cpp/build/bin/whisper-cli.exe"

processor = WhisperProcessor.from_pretrained(
    model_dir,
    language="german",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.eval()

print("=" * 70)
print("W1 HF vs W1 GGML - 10 SAMPLE EQUIVALENCE TEST")
print("=" * 70)

for _, row in df.iterrows():
    audio = row["audio_path"]

    audio_data, sr = sf.read(audio)

    inputs = processor(
        audio_data,
        sampling_rate=sr,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated = model.generate(
            inputs["input_features"],
            language="de",
            task="transcribe",
        )

    hf_text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    result = subprocess.run(
        [
            cli,
            "-m", ggml_model,
            "-f", audio,
            "-l", "de",
            "-nt",
            "-np",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    ggml_text = result.stdout.strip()

    print()
    print(f"[{row['id']}]")
    print(f"REF : {row['t1_reference']}")
    print(f"HF  : {hf_text}")
    print(f"GGML: {ggml_text}")
    print(f"EXIT: {result.returncode}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)