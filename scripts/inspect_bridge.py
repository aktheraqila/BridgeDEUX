"""
BridgeDEUX
Interface Inspection Utility

Purpose
-------
Discovers the current BridgeDEUX interfaces before adding
new infrastructure such as ONNX optimization.

This script DOES NOT modify anything.
It only reports the current project contracts.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig

# Change only if your translator lives elsewhere.
try:
    from models.translators.marian_onnx import MarianONNXTranslator
except ImportError:
    MarianONNXTranslator = None


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_project_config():
    print_header("ProjectConfig")

    ProjectConfig.initialize()

    print("Module:")
    print(ProjectConfig.__module__)

    print("\nPublic Attributes:")

    for name in sorted(dir(ProjectConfig)):
        if name.startswith("_"):
            continue

        value = getattr(ProjectConfig, name)

        if callable(value):
            continue

        print(f"  {name}")


def inspect_onnx_directory():

    print_header("ONNX Directory")

    model_root = Path("models/onnx")

    if not model_root.exists():
        print("models/onnx not found.")
        return

    for path in sorted(model_root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size / (1024 * 1024)
            print(f"{path} ({size:.2f} MB)")


def inspect_subset():

    print_header("Benchmark Dataset")

    candidates = list(Path(".").rglob("*subset*.parquet"))

    if not candidates:
        print("No benchmark subset parquet found.")
        return

    subset = candidates[0]

    print("Dataset:")
    print(subset)

    df = pd.read_parquet(subset)

    print("\nRows:")
    print(len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)


def inspect_translator():

    print_header("MarianONNXTranslator")

    if MarianONNXTranslator is None:
        print("Translator could not be imported.")
        return

    print(inspect.signature(MarianONNXTranslator))

    print("\nPublic Methods:")

    for name, member in inspect.getmembers(
        MarianONNXTranslator,
        inspect.isfunction,
    ):
        if name.startswith("_"):
            continue

        print(name)


def main():

    inspect_project_config()

    inspect_onnx_directory()

    inspect_subset()

    inspect_translator()


if __name__ == "__main__":
    main()