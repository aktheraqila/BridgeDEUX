"""
Phase 2: Verify CoVoST Dataset (Parquet Version)

Purpose
-------
Verify the modern CoVoST dataset before building the
permanent dataset loader.

This script:

1. Downloads the validation split
2. Prints dataset summary
3. Prints available columns
4. Prints one sample
5. Verifies that audio, sentence and translation exist
"""

from datasets import load_dataset


def verify_covost():

    print("=" * 70)
    print("BridgeDEUX - Dataset Verification")
    print("=" * 70)

    print("\nLoading CoVoST 2 (de -> en)...")

    dataset = load_dataset(
        "fixie-ai/covost2",
        "de_en",
        split="validation"
    )

    print("\nDataset Loaded Successfully\n")

    print(dataset)

    print("\nNumber of Samples")
    print("-" * 70)
    print(len(dataset))

    sample = dataset[0]

    print("\nAvailable Fields")
    print("-" * 70)

    for key in sample.keys():
        print(key)

    print("\nSample")
    print("-" * 70)

    print(f"Sentence     : {sample['sentence']}")
    print(f"Translation  : {sample['translation']}")

    if "audio" in sample:

        audio = sample["audio"]

        print("\nAudio")
        print(f"Path         : {audio['path']}")
        print(f"SamplingRate : {audio['sampling_rate']}")
        print(f"Samples      : {len(audio['array'])}")

    print("\nVerification Successful.")

    required = [
        "sentence",
        "translation",
        "audio"
    ]

    missing = []

    for item in required:
        if item not in sample:
            missing.append(item)

    if len(missing) == 0:
        print("\nAll required fields are available.")
    else:
        print("\nMissing fields:")
        print(missing)


if __name__ == "__main__":
    verify_covost()