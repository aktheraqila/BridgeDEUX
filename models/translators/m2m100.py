"""
BridgeDEUX Core Framework
M2M100 Translator Implementation (Frozen v1.0)
"""

from __future__ import annotations

import time
import torch
from typing import Any

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from bridge.logger import BridgeLogger
from models.translators.base_translator import BaseTranslator
from models.translators.result import TranslationResult
from models.translators.exceptions import (
    ModelLoadError,
    TokenizerLoadError,
    TranslationError,
    ModelNotLoadedError
)


class M2M100Translator(BaseTranslator):
    """
    Concrete translator for the multilingual M2M100 architecture.
    Maintains strict compatibility with the BaseTranslator interface while 
    handling internal source/target language routing at runtime.
    """

    def __init__(
        self, 
        source_lang: str = "de",
        target_lang: str = "en",
        model_checkpoint: str = "facebook/m2m100_418M",
        device: str | None = None,
        generation_config: dict[str, Any] | None = None
    ) -> None:
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._checkpoint = model_checkpoint
        self._display_name = "M2M100"
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)
        
        # Explicit Device Validation
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device target: '{requested_device}'. Must be 'cpu' or 'cuda'.")
            
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA execution requested but torch.cuda.is_available() evaluated to False.")
            
        self._device = requested_device
        
        # Serialized Generation Hyperparameters
        self._generation_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 4,
            "early_stopping": True,
        }
        if generation_config:
            self._generation_kwargs.update(generation_config)
        
        self._tokenizer: M2M100Tokenizer | None = None
        self._model: M2M100ForConditionalGeneration | None = None
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return self._display_name

    @property
    def model_version(self) -> str:
        return self._checkpoint

    @property
    def device(self) -> str:
        return self._device

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            self._logger.warning("Load() invoked but execution context is already initialized.")
            return

        self._logger.info("Initializing %s [%s] on hardware: %s (O/S Threads: %d)", 
                          self._display_name, self._checkpoint, self._device, torch.get_num_threads())

        # Atomic step 1: Tokenizer Allocation
        try:
            self._logger.info("Allocating multilingual tokenizer engine...")
            self._tokenizer = M2M100Tokenizer.from_pretrained(
                self._checkpoint,
                src_lang=self._source_lang
            )
        except Exception as e:
            raise TokenizerLoadError(f"Tokenizer initialization failed: {str(e)}") from e

        # Atomic step 2: Model Weights Allocation
        try:
            self._logger.info("Loading model tensor graphs into device memory...")
            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self._checkpoint
            ).to(self._device)

            self._model.eval()

            # Disable the model's default max_length to avoid
            # conflict with BridgeDEUX's max_new_tokens setting.
            self._model.generation_config.max_length = None
        except Exception as e:
            raise ModelLoadError(f"Model tensor graph allocation failed: {str(e)}") from e

        # Post-load validation guard
        if self._tokenizer is None or self._model is None:
            raise ModelLoadError("Translator initialization completed with invalid state.")

        self._is_loaded = True
        self._logger.info("Inference context established successfully.")

    def unload(self) -> None:
        """
        Clears the model and tokenizer from memory to prevent Out-Of-Memory (OOM) 
        errors when switching models during sequential benchmarking.
        """
        if not self._is_loaded:
            return
            
        self._logger.info("Unloading %s from memory...", self._display_name)
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        
        if self._device == "cuda":
            torch.cuda.empty_cache()
            
        self._logger.info("Memory cleared successfully.")

    def translate(self, text: str) -> TranslationResult:
        if not self._is_loaded:
            raise ModelNotLoadedError("Inference request dispatched before engine initialization.")
            
        # Defensive state assertions for static analyzers and runtime integrity
        assert self._tokenizer is not None, "Tokenizer state corrupted (None) despite loaded flag."
        assert self._model is not None, "Model state corrupted (None) despite loaded flag."
        
        # String Normalization & Structural Validation
        clean_text = text.strip()
        if not clean_text:
            raise TranslationError("Execution failed: Input payload cannot be empty or whitespace-only.")

        overall_start = time.perf_counter()

        # Step 1: Tokenization Phase
        tok_start = time.perf_counter()
        
        # Critical M2M100 Configuration: Set the source language state right before encoding
        self._tokenizer.src_lang = self._source_lang
        
        inputs = self._tokenizer(
            clean_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Explicit tensor routing to hardware target
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        
        time_tok = (time.perf_counter() - tok_start) * 1000

        # Step 2: Auto-Regressive Tensor Generation Phase
        gen_start = time.perf_counter()
        try:
            # Critical M2M100 Configuration: Retrieve target language ID for generation
            forced_bos_token_id = self._tokenizer.get_lang_id(self._target_lang)
            
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    **self._generation_kwargs
                )
        except Exception as e:
            raise TranslationError(f"Auto-regressive tensor generation encountered runtime failure: {str(e)}") from e
        time_gen = (time.perf_counter() - gen_start) * 1000

        # Step 3: Decoding Phase
        dec_start = time.perf_counter()
        translation = self._tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        time_dec = (time.perf_counter() - dec_start) * 1000
        
        # Output Validation
        if not translation.strip():
            raise TranslationError("Safety Guard: Model emitted an empty string or illegal null tokens.")

        overall_end = time.perf_counter()
        
        return TranslationResult(
            model_name=self.model_name,
            model_version=self.model_version,
            source_text=clean_text,
            translation=translation,
            input_tokens=inputs["input_ids"].shape[1],
            output_tokens=generated_tokens.shape[1],
            tokenization_time_ms=time_tok,
            generation_time_ms=time_gen,
            decoding_time_ms=time_dec,
            total_time_ms=(overall_end - overall_start) * 1000
        )

    def translate_batch(self, texts: list[str]) -> list[TranslationResult]:
        raise NotImplementedError("Batch translation has not yet been implemented.")