#!/usr/bin/env python3
"""
BridgeDEUX: Profile German Token Coverage for Vocabulary Pruning
==============================================================
Scans CoVoST2 German text to determine the minimum Whisper token set
required to achieve 99.0%, 99.9%, and 100.0% corpus coverage.
"""

import sys
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
from tqdm import tqdm
from transformers import WhisperTokenizer

from bridge.config import ProjectConfig
from datasets.providers.covost_provider import CoVoSTProvider


def main():
    ProjectConfig.initialize()
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-base", language="de", task="transcribe"
    )

    print("=" * 70)
    print(" PROFILING GERMAN TOKEN FREQUENCY IN COVOST2")
    print("=" * 70)

    # 1. Collect all German transcript text across test & train splits
    sentences = []
    for split in ["test", "train"]:
        try:
            provider = CoVoSTProvider(split=split, include_audio=False)
            for sample in provider:
                sentences.append(sample.source_text)
        except Exception as e:
            print(f"[*] Note: Split '{split}' skipped or unavailable: {e}")

    print(f"Total Sentences Collected: {len(sentences):,}")

    # 2. Tokenize and count token frequencies
    token_counts = Counter()
    for text in tqdm(sentences, desc="Tokenizing"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(ids)

    # 3. Always preserve special tokens and language flags
    all_special_ids = set(tokenizer.all_special_ids)
    observed_token_ids = set(token_counts.keys())

    total_tokens_seen = sum(token_counts.values())
    unique_tokens_seen = len(observed_token_ids)

    print("-" * 70)
    print(f"Total Subword Tokens Processed : {total_tokens_seen:,}")
    print(f"Unique German Subwords Seen   : {unique_tokens_seen:,} / 51,865")
    print(f"Preserved Special Tokens      : {len(all_special_ids)}")

    # 4. Calculate coverage thresholds
    sorted_tokens = token_counts.most_common()
    cumulative = 0
    coverage_cutoffs = {0.99: None, 0.999: None, 1.0: unique_tokens_seen}

    for rank, (tid, count) in enumerate(sorted_tokens, start=1):
        cumulative += count
        ratio = cumulative / total_tokens_seen
        for cov in [0.99, 0.999]:
            if ratio >= cov and coverage_cutoffs[cov] is None:
                coverage_cutoffs[cov] = rank

    print("-" * 70)
    print(" VOCABULARY SIZE VS. CORPUS COVERAGE")
    print("-" * 70)
    for cov, count in coverage_cutoffs.items():
        total_vocab = count + len(all_special_ids - set(dict(sorted_tokens[:count]).keys()))
        saved_params = (51865 - total_vocab) * 512
        pct_reduction = (saved_params / 72.59e6) * 100
        print(
            f"Coverage {cov*100:5.1f}% -> Vocab: {total_vocab:5d} tokens | "
            f"Param Drop: {saved_params / 1e6:5.2f}M (-{pct_reduction:.1f}% total model)"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()