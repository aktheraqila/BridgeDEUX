#!/usr/bin/env python3

import logging
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("MathVerifier")


def shift_right(labels, pad_token_id, decoder_start_token_id):
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id

    # HF convention: -100 labels become PAD in decoder inputs.
    shifted.masked_fill_(shifted == -100, pad_token_id)

    return shifted


def main():

    model_name = "Helsinki-NLP/opus-mt-de-en"

    logger.info("Loading PyTorch MarianMT...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    model.eval()

    source = "Mückenstiche sollte man nicht aufkratzen."
    target = "You shouldn't scratch mosquito bites."

    inputs = tokenizer(source, return_tensors="pt")

    labels = tokenizer(
        text_target=target,
        return_tensors="pt"
    )["input_ids"]

    # ---------------------------------------------------------
    # 1. Hugging Face's built-in loss
    # ---------------------------------------------------------

    with torch.no_grad():

        official = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

    official_loss = official.loss.item()

    # ---------------------------------------------------------
    # 2. Reproduce the exact decoder-input construction
    # ---------------------------------------------------------

    decoder_input_ids = shift_right(
        labels,
        model.config.pad_token_id,
        model.config.decoder_start_token_id
    )

    # ---------------------------------------------------------
    # 3. Explicit forward pass
    # ---------------------------------------------------------

    with torch.no_grad():

        explicit = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )

    logits = explicit.logits

    # ---------------------------------------------------------
    # 4. Manual cross entropy
    # ---------------------------------------------------------

    vocab_size = logits.shape[-1]

    loss_fct = torch.nn.CrossEntropyLoss(
        ignore_index=-100
    )

    manual_loss = loss_fct(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1)
    ).item()

    # ---------------------------------------------------------
    # 5. Compare
    # ---------------------------------------------------------

    difference = abs(official_loss - manual_loss)

    print("\n" + "=" * 65)
    print(" MARIAN LOSS MATHEMATICS VERIFICATION")
    print("=" * 65)

    print(f"Official HF loss : {official_loss:.8f}")
    print(f"Manual loss      : {manual_loss:.8f}")
    print(f"Absolute diff    : {difference:.10f}")

    print(f"\nLogits shape     : {tuple(logits.shape)}")
    print(f"Labels shape     : {tuple(labels.shape)}")
    print(f"Decoder inputs   : {tuple(decoder_input_ids.shape)}")

    print("-" * 65)

    if difference < 1e-5:

        print("[PASS]")
        print(
            "Manual loss matches Hugging Face's loss "
            "using the same shifted decoder inputs."
        )

    else:

        print("[FAIL]")
        print(
            "Manual loss does not match Hugging Face's "
            "built-in loss."
        )

    print("=" * 65)


if __name__ == "__main__":
    main()