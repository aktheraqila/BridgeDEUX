from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ZIP_ROOT = "MSLT_Corpus/Data/MSLT_Test_DE_20160516"
ID_PATTERN = re.compile(r"MSLT_Test_DE_(\d{4})\.T0\.de\.wav$")


def read_snt(archive: zipfile.ZipFile, member: str) -> str:
    return archive.read(member).decode("utf-16").strip()


def build_mslt_asr_cache(zip_path: Path) -> None:
    project_root = Path.cwd()

    raw_dir = (
        project_root
        / "datasets"
        / "raw"
        / "mslt"
        / "de_en"
        / "test"
    )

    cache_dir = (
        project_root
        / "datasets"
        / "cache"
        / "mslt"
        / "de_en"
        / "test"
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cache_dir / "mslt_de_asr_test.parquet"
    csv_path = cache_dir / "mslt_de_asr_test.csv"
    metadata_path = cache_dir / "mslt_de_asr_test.json"

    records = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = set(archive.namelist())

        t0_members = sorted(
            name
            for name in members
            if name.startswith(ZIP_ROOT + "/")
            and name.endswith(".T0.de.wav")
        )

        print(f"T0 audio discovered: {len(t0_members)}")

        if len(t0_members) != 2275:
            raise RuntimeError(
                f"Expected 2275 German Test T0 files, "
                f"found {len(t0_members)}"
            )

        for t0_member in t0_members:
            match = ID_PATTERN.search(t0_member)

            if not match:
                raise RuntimeError(
                    f"Could not extract utterance ID from: {t0_member}"
                )

            utterance_id = match.group(1)

            prefix = f"{ZIP_ROOT}/MSLT_Test_DE_{utterance_id}"

            t1_member = prefix + ".T1.de.snt"
            t2_member = prefix + ".T2.de.snt"

            if t1_member not in members:
                raise RuntimeError(f"Missing T1: {utterance_id}")

            if t2_member not in members:
                raise RuntimeError(f"Missing T2: {utterance_id}")

            t1 = read_snt(archive, t1_member)
            t2 = read_snt(archive, t2_member)

            filename = f"MSLT_Test_DE_{utterance_id}.T0.de.wav"
            audio_path = raw_dir / filename

            if not audio_path.exists():
                with archive.open(t0_member) as source:
                    with open(audio_path, "wb") as target:
                        shutil.copyfileobj(source, target)

            relative_audio = audio_path.relative_to(project_root)

            records.append(
                {
                    "id": utterance_id,
                    "audio_path": str(relative_audio).replace("\\", "/"),
                    "t1_reference": t1,
                    "t2_reference": t2,
                    "dataset": "MSLT",
                    "split": "test",
                    "language": "de",
                }
            )

    dataframe = pd.DataFrame(records)

    if len(dataframe) != 2275:
        raise RuntimeError(
            f"Expected 2275 records, created {len(dataframe)}"
        )

    if dataframe["id"].duplicated().any():
        raise RuntimeError("Duplicate IDs detected.")

    if dataframe["audio_path"].isnull().any():
        raise RuntimeError("Null audio_path detected.")

    if dataframe["t1_reference"].isnull().any():
        raise RuntimeError("Null T1 reference detected.")

    dataframe.to_parquet(parquet_path, index=False)
    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "dataset": "MSLT",
        "split": "test",
        "language": "de",
        "row_count": len(dataframe),
        "audio_count": len(dataframe),
        "empty_t1": int((dataframe["t1_reference"] == "").sum()),
        "empty_t2": int((dataframe["t2_reference"] == "").sum()),
        "columns": list(dataframe.columns),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("MSLT ASR CACHE COMPLETE")
    print(f"Records       : {len(dataframe)}")
    print(f"Audio files   : {len(dataframe)}")
    print(f"Empty T1      : {metadata['empty_t1']}")
    print(f"Empty T2      : {metadata['empty_t2']}")
    print(f"Parquet       : {parquet_path}")
    print(f"CSV           : {csv_path}")
    print(f"Metadata      : {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip_path",
        type=Path,
        default=Path(
            r"C:\Users\user\Downloads\MSLT_Corpus.zip"
        ),
    )

    args = parser.parse_args()

    build_mslt_asr_cache(args.zip_path)


if __name__ == "__main__":
    main()