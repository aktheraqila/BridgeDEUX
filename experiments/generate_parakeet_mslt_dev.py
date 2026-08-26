import json
import subprocess
from pathlib import Path

from bridge.config import ProjectConfig
from datasets.providers.mslt_provider import MSLTProvider


MODEL = Path("models/parakeet/tdt-0.6b-v3-f16.gguf")
PARAKEET = Path("models/parakeet/parakeet-cli.exe")

TRAIN_IDS = Path("datasets/manifests/mslt_dev_train_ids.txt")
VAL_IDS = Path("datasets/manifests/mslt_dev_val_ids.txt")

OUTPUT = Path(
    "datasets/cache/mslt/de_en/dev/parakeet_dev_predictions.jsonl"
)


def run_parakeet(audio_path: Path):
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
        return {
            "text": "",
            "error": result.stderr.strip(),
        }

    try:
        data = json.loads(result.stdout)

        return {
            "text": data.get("text", ""),
            "raw_json": data,
            "error": None,
        }

    except json.JSONDecodeError:
        return {
            "text": "",
            "raw_json": None,
            "error": "Invalid JSON: " + result.stdout.strip(),
        }


def main():
    ProjectConfig.initialize()

    train_ids = {
        x.strip()
        for x in TRAIN_IDS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    val_ids = {
        x.strip()
        for x in VAL_IDS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    expected_ids = train_ids | val_ids

    provider = MSLTProvider(
        split="dev",
        include_audio=False,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    completed = set()

    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed.add(
                            json.loads(line)["sample_id"]
                        )
                    except Exception:
                        pass

    print("Dev samples :", len(provider))
    print("Expected    :", len(expected_ids))
    print("Completed   :", len(completed))
    print("Remaining   :", len(expected_ids - completed))
    print("Output      :", OUTPUT)

    with OUTPUT.open("a", encoding="utf-8") as out:

        for i in range(len(provider)):
            sample = provider[i]

            sid = str(sample.id).strip()

            if sid not in expected_ids:
                continue

            if sid in completed:
                continue

            audio_path = Path(sample.file_name)

            result = run_parakeet(audio_path)

            row = {
                "sample_id": sid,
                "audio_path": str(audio_path.resolve()),
                "reference": sample.source_text,
                "parakeet_transcript": result["text"],
                "parakeet_raw_json": result["raw_json"],
                "error": result["error"],
                "split": "dev",
                "subset": (
                    "validation"
                    if sid in val_ids
                    else "train"
                ),
            }

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            completed.add(sid)

            print(
                f"[{len(completed)}/{len(expected_ids)}] "
                f"{sid} | {result['text']}"
            )

    print("\nDONE")
    print("Predictions:", len(completed))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()