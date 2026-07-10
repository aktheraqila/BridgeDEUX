"""
Experiment Logger

Purpose
-------
Centralized experiment logging.

This module is responsible ONLY for saving experiment results.
It does not know anything about Marian, M2M100, NLLB,
CoVoST, Android, or ONNX.

Every experiment in the thesis should eventually use this logger.
"""

import csv
from datetime import datetime
from pathlib import Path


class ExperimentLogger:

    def __init__(self):

        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.csv_file = self.results_dir / "mt_validation.csv"

        self.headers = [
            "Timestamp",
            "Experiment ID",
            "Model",
            "Direction",
            "Framework",
            "Device",

            "Tokenizer Load (s)",
            "Model Load (s)",
            "Tokenization (s)",
            "Generation (s)",
            "Decoding (s)",
            "Total (s)",

            "Notes"
        ]

        if not self.csv_file.exists():

            with open(
                self.csv_file,
                mode="w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)
                writer.writerow(self.headers)

    def log(self, result: dict):

        row = [

            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            result.get("experiment_id", ""),

            result["model_name"],

            result.get("direction", ""),

            result.get("framework", "PyTorch"),

            result.get("device", "Desktop"),

            f"{result['timings']['tokenizer_load']:.4f}",
            f"{result['timings']['model_load']:.4f}",
            f"{result['timings']['tokenization']:.4f}",
            f"{result['timings']['generation']:.4f}",
            f"{result['timings']['decoding']:.4f}",
            f"{result['timings']['total']:.4f}",

            result.get("notes", "")
        ]

        with open(
            self.csv_file,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(row)

        print(f"Results saved to {self.csv_file}")