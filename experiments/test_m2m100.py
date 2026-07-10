"""
BridgeDEUX Core Framework
Verification Script for M2M100Translator Implementation (Version 1.0 - Frozen)
"""

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from models.translators.m2m100 import M2M100Translator
from models.translators.exceptions import (
    TranslatorError, 
    ModelLoadError, 
    TokenizerLoadError, 
    TranslationError, 
    ModelNotLoadedError
)


def main():
    # Initialize Core Framework Configuration
    ProjectConfig.initialize()
    logger = BridgeLogger.get_logger("TestM2M100")
    
    print("\n" + "="*60)
    print(" BRIDGE-DEUX: M2M100 TRANSLATOR VERIFICATION SUITE")
    print("="*60)
    logger.info("Starting M2M100Translator validation suite.")

    # Track verification metrics for thesis summary reporting
    passed_tests = 0
    failed_tests = 0
    
    # Instantiate the translator using default settings (DE -> EN)
    translator = M2M100Translator(source_lang="de", target_lang="en")

    # ---------------------------------------------------------
    # TEST 1: Unloaded Model Guard
    # ---------------------------------------------------------
    print("\n[Test 1] Guard: ModelNotLoadedError")
    logger.info("Executing Test 1: Unloaded model guard check.")
    try:
        translator.translate("Guten Tag.")
        print("❌ FAILED: Did not raise ModelNotLoadedError.")
        logger.error("Test 1 FAILED: Engine allowed inference before load().")
        failed_tests += 1
    except ModelNotLoadedError:
        print("✅ PASSED: Guard successfully blocked unloaded inference.")
        logger.info("Test 1 PASSED: ModelNotLoadedError raised correctly.")
        passed_tests += 1

    # ---------------------------------------------------------
    # TEST 2A: Model Loading Execution
    # ---------------------------------------------------------
    print("\n[Test 2A] Execution: Target Model Initialization")
    logger.info("Executing Test 2A: Core model loading.")
    try:
        translator.load()
        
        # Rigorous State Verification (Optimization-Safe)
        if translator.is_loaded():
            print(f"✅ PASSED: Model [{translator.model_version()}] allocated on {translator.device().upper()} and reports loaded state.")
            logger.info(f"Test 2A PASSED: Model loaded on device {translator.device()}.")
            passed_tests += 1
        else:
            print("❌ FAILED: Translator reports unloaded state after load().")
            logger.error("Test 2A FAILED: Translator internal state reflects uninitialized status post-load execution.")
            failed_tests += 1
            
    except (ModelLoadError, TokenizerLoadError) as e:
        print(f"❌ FAILED: Subsystem initialization exception: {e}")
        logger.error(f"Test 2A FAILED: Target initialization exception: {str(e)}")
        failed_tests += 1
        return
    except TranslatorError as e:
        print(f"❌ FAILED: Generic translator exception during initialization: {e}")
        logger.error(f"Test 2A FAILED: Generic translator exception: {str(e)}")
        failed_tests += 1
        return
    except Exception as e:
        print(f"❌ FAILED: Unexpected critical error during load: {e}")
        logger.exception("Test 2A CRITICAL FAILURE: Non-framework error intercepted during load.")
        failed_tests += 1
        raise

    # ---------------------------------------------------------
    # TEST 2B: Idempotent Load Safety
    # ---------------------------------------------------------
    print("\n[Test 2B] Guard: Idempotent Load Lifecycle")
    logger.info("Executing Test 2B: Idempotency check for repeated load() calls.")
    try:
        translator.load()
        print("✅ PASSED: Repeated load() caught defensively without crashing.")
        logger.info("Test 2B PASSED: Engine handled redundant initialization gracefully.")
        passed_tests += 1
    except TranslatorError as e:
        print(f"❌ FAILED: Repeated load call triggered unexpected TranslatorError: {e}")
        logger.error(f"Test 2B FAILED: Idempotent TranslatorError: {str(e)}")
        failed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Unexpected infrastructure error during repeated load: {e}")
        logger.exception("Test 2B CRITICAL FAILURE: Unhandled non-framework exception on repeated load.")
        failed_tests += 1
        raise

    # ---------------------------------------------------------
    # TEST 3: Standard Inference & Metrics Evaluation
    # ---------------------------------------------------------
    print("\n[Test 3] Execution: Translation & Metric Validation")
    logger.info("Executing Test 3: Active inference profiling and metric verification.")
    test_sentence = "Zieht euch bitte draußen die Schuhe aus."
    
    try:
        result = translator.translate(test_sentence)
        print(f"   Source : {result.source_text}")
        print(f"   Target : {result.translation}")
        
        logger.info("Test 3 execution profiles generated: In_Tokens=%d, Out_Tokens=%d, Latency=%.2fms", 
                    result.input_tokens, result.output_tokens, result.total_time_ms)
        
        # Verify Metadata Invariants
        if (result.model_name == translator.model_name() and 
                result.model_version == translator.model_version()):
            print("✅ PASSED: Model self-describing metadata correctly populated.")
            logger.info(f"Metadata verified: Name={result.model_name}, Version={result.model_version}")
            passed_tests += 1
        else:
            print("❌ FAILED: Metadata mismatch between TranslationResult and Translator instances.")
            logger.error("Metadata verification failure detected.")
            failed_tests += 1

        # Verify Token Integrity
        if result.input_tokens > 0 and result.output_tokens > 0:
            print(f"✅ PASSED: Token counts populated (In: {result.input_tokens}, Out: {result.output_tokens})")
            passed_tests += 1
        else:
            print("❌ FAILED: Zero or negative token counts recorded.")
            logger.error("Token verification failure detected.")
            failed_tests += 1

        # Verify Positive Phase Timing Signatures
        timings = [
            result.tokenization_time_ms, 
            result.generation_time_ms, 
            result.decoding_time_ms
        ]
        if all(t > 0 for t in timings) and result.total_time_ms > 0:
            print(f"✅ PASSED: Latency metric profiles are valid (>0ms).")
            print(f"     -> Tokenization : {result.tokenization_time_ms:.2f} ms")
            print(f"     -> Generation   : {result.generation_time_ms:.2f} ms")
            print(f"     -> Decoding     : {result.decoding_time_ms:.2f} ms")
            print(f"     -> Total Latency: {result.total_time_ms:.2f} ms")
            passed_tests += 1
        else:
            print("❌ FAILED: Invalid negative values detected within latency metrics.")
            logger.error("Timing verification checks failed due to non-positive parameters.")
            failed_tests += 1

        # Real Wall-Clock Verification Check with Tolerance
        EPSILON_MS = 1.0
        summed_phases = result.tokenization_time_ms + result.generation_time_ms + result.decoding_time_ms
        if result.total_time_ms + EPSILON_MS >= summed_phases:
            print("✅ PASSED: Total wall-clock runtime validation inequality holds true.")
            logger.info("Wall-clock check passed: End-to-end latency encapsulates step sum.")
            passed_tests += 1
        else:
            print(f"❌ FAILED: Timing discrepancy. Total ({result.total_time_ms:.2f}ms) + Epsilon < Summed Phases ({summed_phases:.2f}ms).")
            logger.error("Wall-clock verification validation failure.")
            failed_tests += 1

    except TranslationError as e:
        print(f"❌ FAILED: Core model runtime generation failure: {e}")
        logger.error(f"Test 3 FAILED: Inference path exception: {str(e)}")
        failed_tests += 1

    # ---------------------------------------------------------
    # TEST 4: Empty Input Validation Guard
    # ---------------------------------------------------------
    print("\n[Test 4] Guard: Empty Payload Handling")
    logger.info("Executing Test 4: Empty/Whitespace structural input guard check.")
    try:
        translator.translate("   \n  ")
        print("❌ FAILED: Empty whitespace payload bypassed translator validation checks.")
        logger.error("Test 4 FAILED: Empty payload allowed down the parsing pipeline.")
        failed_tests += 1
    except TranslationError:
        print("✅ PASSED: Structural guard blocked empty text payload string execution.")
        logger.info("Test 4 PASSED: TranslationError thrown safely on corrupt structural data input.")
        passed_tests += 1

    # ---------------------------------------------------------
    # SUMMARY REPORT & AUTOMATION EXIT
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" VERIFICATION SUITE COMPLETE")
    print(f" PASSED: {passed_tests}")
    print(f" FAILED: {failed_tests}")
    print("="*60 + "\n")
    
    logger.info(f"Validation complete. Summary metrics -> Passed: {passed_tests}, Failed: {failed_tests}")

    # Process exit code for automation frameworks
    if failed_tests > 0:
        raise SystemExit(1)
    
    raise SystemExit(0)


if __name__ == "__main__":
    main()