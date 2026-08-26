import csv
import subprocess
from pathlib import Path

from bridge.config import ProjectConfig
from datasets.providers.mslt_provider import MSLTProvider


MODEL = Path("models/parakeet/tdt-0.6b-v3-f16.gguf")
PARAKEET = Path("models/parakeet/parakeet-cli.exe")

OUTPUT_CSV = Path("experiments/results/parakeet_mslt_diagnostic.csv")

NUM_SAMPLES = 10


def run_parakeet(audio_path: Path) -> str:
    command = [
        str(PARAKEET),
        "transcribe",
        "--model",
        str(MODEL),
        "--input",
        str(audio_path),
        "--decoder",
        "tdt",
        "--json",
    ]

    result = subprocess.run(
    command,
    capture_output=True,
    text=True,
    encoding="utf-8",
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"

    return result.stdout.strip()


def main():
    ProjectConfig.initialize()

    provider = MSLTProvider(
        split="test",
        include_audio=False,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    limit = min(NUM_SAMPLES, len(provider))

    for i in range(limit):
        sample = provider[i]

        audio_path = Path(sample.file_name)

        prediction = run_parakeet(audio_path)

        rows.append(
            {
                "id": sample.id,
                "reference": sample.source_text,
                "prediction_json": prediction,
                "audio": str(audio_path),
            }
        )

        print(f"\n[{i + 1}/{limit}]")
        print("ID        :", sample.id)
        print("Reference :", sample.source_text)
        print("Prediction:", prediction)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "reference",
                "prediction_json",
                "audio",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()