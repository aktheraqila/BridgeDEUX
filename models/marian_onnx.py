"""
BridgeDEUX Core Framework
MarianMT ONNX Translator Implementation (Refined v1.1)
"""

import time
import gc
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer

from bridge.logger import BridgeLogger
from models.translators.base_translator import BaseTranslator
from models.translators.result import TranslationResult
from models.translators.exceptions import (
    ModelLoadError,
    TokenizerLoadError,
    TranslationError,
    ModelNotLoadedError
)

class MarianONNXTranslator(BaseTranslator):
    """
    ONNX Runtime concrete implementation for MarianMT architectures.
    Wraps ORTModelForSeq2SeqLM to provide an identical execution pipeline 
    to the PyTorch baseline, enabling strict A/B benchmarking.
    """

    def __init__(
        self, 
        onnx_model_dir: str | Path = "models/onnx/opus_mt_de_en",
        provider: str = "CPUExecutionProvider"
    ) -> None:
        self._checkpoint = "Helsinki-NLP/opus-mt-de-en"
        self._onnx_model_dir = Path(onnx_model_dir)
        self._display_name = "MarianMT"
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)
        
        self._provider = provider
        
        # TODO: Centralize generation kwargs into a shared configuration object 
        # so both PyTorch and ONNX backends consume the exact same source.
        self._generation_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 4,
            "early_stopping": True,
        }
            
        self._tokenizer: MarianTokenizer | None = None
        self._model: ORTModelForSeq2SeqLM | None = None
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return self._display_name

    @property
    def model_version(self) -> str:
        return f"{self._checkpoint} (ONNX)"

    @property
    def device(self) -> str:
        return self._provider

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            return

        self._logger.info("Initializing %s [%s]", self._display_name, self.model_version)

        if not self._onnx_model_dir.exists():
            raise ModelLoadError(f"ONNX artifacts not found at: {self._onnx_model_dir}")

        try:
            self._logger.info("Allocating ONNX tokenizer engine...")
            self._tokenizer = MarianTokenizer.from_pretrained(self._onnx_model_dir)
        except Exception as e:
            raise TokenizerLoadError(f"Tokenizer initialization failed: {e}") from e

        try:
            self._logger.info("Loading ONNX computation graphs into memory...")
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                self._onnx_model_dir, 
                provider=self._provider,
                use_cache=True
            )
            
            if hasattr(self._model, "generation_config"):
                self._model.generation_config.max_length = None
                
        except Exception as e:
            raise ModelLoadError(f"ONNX graph allocation failed: {e}") from e

        self._is_loaded = True
        self._logger.info("ONNX Inference context established successfully.")
        
        active_providers = getattr(self._model, "providers", [self._provider])
        self._logger.info("Execution Providers Active: %s", active_providers)

    def unload(self) -> None:
        if not self._is_loaded:
            return
            
        self._logger.info("Unloading %s from memory...", self.model_version)
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        
        # Request Python garbage collection after releasing model references
        gc.collect()
        self._logger.info("Garbage collection requested.")

    def translate(self, text: str) -> TranslationResult:
        if not self._is_loaded:
            raise ModelNotLoadedError("Inference request dispatched before engine initialization.")
            self._is_warmed_up = True
            
        if self._tokenizer is None:
            raise TokenizerLoadError("Tokenizer state corrupted.")
        if self._model is None:
            raise ModelLoadError("ONNX Model state corrupted.")
        
        clean_text = text.strip()
        if not clean_text:
            raise TranslationError("Input payload cannot be empty or whitespace-only.")

        overall_start = time.perf_counter()

        tok_start = time.perf_counter()
        inputs = self._tokenizer(
            clean_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        time_tok = (time.perf_counter() - tok_start) * 1000

        gen_start = time.perf_counter()
        try:
            generated_tokens = self._model.generate(**inputs, **self._generation_kwargs)
        except Exception as e:
            raise TranslationError(f"ONNX generation encountered runtime failure: {e}") from e
        time_gen = (time.perf_counter() - gen_start) * 1000

        dec_start = time.perf_counter()
        translation = self._tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        time_dec = (time.perf_counter() - dec_start) * 1000

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
        """
        Supports efficient batched generation. 
        Note: Latency metrics returned in these TranslationResults are batch-averages, 
        not true per-sample measurements, due to sequence padding.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError("Inference request dispatched before engine initialization.")
            
        if self._tokenizer is None or self._model is None:
            raise ModelLoadError("Model state corrupted.")
            
        clean_texts = [t.strip() for t in texts if t.strip()]
        if not clean_texts:
            raise TranslationError("Batch cannot be empty.")
            
        overall_start = time.perf_counter()
        
        tok_start = time.perf_counter()
        inputs = self._tokenizer(
            clean_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        time_tok = (time.perf_counter() - tok_start) * 1000
        
        gen_start = time.perf_counter()
        try:
            generated_tokens = self._model.generate(**inputs, **self._generation_kwargs)
        except Exception as e:
            raise TranslationError(f"ONNX batch generation failed: {e}") from e
        time_gen = (time.perf_counter() - gen_start) * 1000
        
        dec_start = time.perf_counter()
        translations = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        time_dec = (time.perf_counter() - dec_start) * 1000
        overall_end = time.perf_counter()
        
        avg_tok = time_tok / len(clean_texts)
        avg_gen = time_gen / len(clean_texts)
        avg_dec = time_dec / len(clean_texts)
        avg_tot = ((overall_end - overall_start) * 1000) / len(clean_texts)
        
        results = []
        for i, (src, tgt) in enumerate(zip(clean_texts, translations)):
            in_toks = (inputs["input_ids"][i] != self._tokenizer.pad_token_id).sum().item()
            out_toks = (generated_tokens[i] != self._tokenizer.pad_token_id).sum().item()
            
            results.append(TranslationResult(
                model_name=self.model_name,
                model_version=self.model_version,
                source_text=src,
                translation=tgt,
                input_tokens=in_toks,
                output_tokens=out_toks,
                tokenization_time_ms=avg_tok,
                generation_time_ms=avg_gen,
                decoding_time_ms=avg_dec,
                total_time_ms=avg_tot
            ))
            
        return results