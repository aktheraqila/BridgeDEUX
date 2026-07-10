"""
Inspect the Hugging Face generation configuration
used by MarianMT.
"""

from bridge.config import ProjectConfig
from models.translators.marian import MarianTranslator


def main():

    ProjectConfig.initialize()

    translator = MarianTranslator()

    translator.load()

    try:
        config = translator._model.generation_config

        print("\n===== Generation Config =====")
        print(config)
        print("=============================\n")

        print(f"max_length      : {config.max_length}")
        print(f"max_new_tokens  : {config.max_new_tokens}")
        print(f"min_length      : {config.min_length}")
        print(f"num_beams       : {config.num_beams}")

    finally:
        translator.unload()


if __name__ == "__main__":
    main()