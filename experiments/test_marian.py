"""
BridgeDEUX Core Framework
Verification Script for MarianMT Implementation (Version 1.0)
"""

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from models.translators.marian import MarianTranslator
from models.translators.exceptions import (
    TranslatorError,
    ModelLoadError,
    TokenizerLoadError,
    TranslationError,
    ModelNotLoadedError,
)


def main() -> None:
    ProjectConfig.initialize()

    logger = BridgeLogger.get_logger("TestMarian")

    passed_tests = 0
    failed_tests = 0

    WALL_CLOCK_EPSILON_MS = 1.0

    print("\n" + "=" * 70)
    print(" BRIDGEDEUX - MARIAN TRANSLATOR VERIFICATION SUITE ")
    print("=" * 70)

    logger.info("Starting MarianTranslator verification suite.")

    translator = MarianTranslator()

    # ==========================================================
    # Test 1
    # ==========================================================

    print("\n[Test 1] Unloaded Model Guard")
    logger.info("Running Test 1.")

    try:
        translator.translate("Guten Tag.")

        print("❌ FAILED")
        print("Model allowed inference before load().")

        logger.error("Model allowed inference before load().")
        failed_tests += 1

    except ModelNotLoadedError:

        print("✅ PASSED")

        logger.info("ModelNotLoadedError raised correctly.")
        passed_tests += 1

    # ==========================================================
    # Test 2A
    # ==========================================================

    print("\n[Test 2A] Model Loading")
    logger.info("Running Test 2A.")

    try:
        translator.load()

        if translator.is_loaded():

            print("✅ PASSED")
            print(f"Model      : {translator.model_name()}")
            print(f"Checkpoint : {translator.model_version()}")
            print(f"Device     : {translator.device()}")

            logger.info("Model loaded successfully.")
            passed_tests += 1

        else:

            print("❌ FAILED")
            print("Translator reports unloaded state after load().")

            logger.error("Translator not loaded after load().")
            failed_tests += 1

    except (ModelLoadError, TokenizerLoadError) as e:

        print("❌ FAILED")
        print(e)

        logger.exception("Initialization failed.")
        failed_tests += 1

    except TranslatorError as e:

        print("❌ FAILED")
        print(e)

        logger.exception("Translator initialization failed.")
        failed_tests += 1

    # ==========================================================
    # Test 2B
    # ==========================================================

    print("\n[Test 2B] Idempotent Load")
    logger.info("Running Test 2B.")

    try:

        translator.load()

        print("✅ PASSED")
        print("Repeated load() handled correctly.")

        logger.info("Repeated load() handled correctly.")
        passed_tests += 1

    except TranslatorError as e:

        print("❌ FAILED")
        print(e)

        logger.exception("Repeated load() failed.")
        failed_tests += 1

    # ==========================================================
    # Test 3
    # ==========================================================

    print("\n[Test 3] Translation")
    logger.info("Running Test 3.")

    sentence = "Zieht euch bitte draußen die Schuhe aus."

    try:

        result = translator.translate(sentence)

        print("\nSource")
        print(result.source_text)

        print("\nTranslation")
        print(result.translation)

        # ------------------------------------------------------
        # Metadata
        # ------------------------------------------------------

        if (
            result.model_name == translator.model_name()
            and result.model_version == translator.model_version()
        ):

            print("\n✅ Metadata populated correctly.")

        else:

            print("\n❌ Metadata mismatch.")

        # ------------------------------------------------------
        # Token Counts
        # ------------------------------------------------------

        if result.input_tokens > 0 and result.output_tokens > 0:

            print(
                f"✅ Token Counts : "
                f"Input={result.input_tokens}, "
                f"Output={result.output_tokens}"
            )

        else:

            print("❌ Invalid token counts.")

        # ------------------------------------------------------
        # Timing
        # ------------------------------------------------------

        print("\nTiming")

        print(f"Tokenization : {result.tokenization_time_ms:.2f} ms")
        print(f"Generation   : {result.generation_time_ms:.2f} ms")
        print(f"Decoding     : {result.decoding_time_ms:.2f} ms")
        print(f"Total        : {result.total_time_ms:.2f} ms")

        if (
            result.tokenization_time_ms > 0
            and result.generation_time_ms > 0
            and result.decoding_time_ms > 0
            and result.total_time_ms > 0
        ):

            print("✅ Positive timing values.")

        else:

            print("❌ Invalid timing values.")

        summed = (
            result.tokenization_time_ms
            + result.generation_time_ms
            + result.decoding_time_ms
        )

        if result.total_time_ms + WALL_CLOCK_EPSILON_MS >= summed:

            print("✅ Wall-clock validation passed.")

        else:

            print("❌ Wall-clock validation failed.")

        logger.info(
            "Inference complete. "
            f"InputTokens={result.input_tokens}, "
            f"OutputTokens={result.output_tokens}, "
            f"Latency={result.total_time_ms:.2f} ms"
        )

        passed_tests += 1

    except TranslationError as e:

        print("❌ FAILED")
        print(e)

        logger.exception("Translation failed.")
        failed_tests += 1

    # ==========================================================
    # Test 4
    # ==========================================================

    print("\n[Test 4] Empty Input Guard")
    logger.info("Running Test 4.")

    try:

        translator.translate("      ")

        print("❌ FAILED")
        print("Empty input accepted.")

        logger.error("Empty input accepted.")
        failed_tests += 1

    except TranslationError:

        print("✅ PASSED")
        print("Empty input rejected correctly.")

        logger.info("Empty input rejected correctly.")
        passed_tests += 1

    # ==========================================================
    # Summary
    # ==========================================================

    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY ")
    print("=" * 70)

    print(f"Passed : {passed_tests}")
    print(f"Failed : {failed_tests}")

    logger.info(
        f"Verification complete. Passed={passed_tests}, Failed={failed_tests}"
    )

    if failed_tests > 0:
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()