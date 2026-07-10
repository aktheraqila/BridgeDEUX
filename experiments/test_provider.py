from bridge.config import ProjectConfig
from datasets.providers.covost_provider import CoVoSTProvider


def main():
    ProjectConfig.initialize()

    provider = CoVoSTProvider()

    print(f"Samples : {len(provider)}")

    sample = provider[0]

    print("\nFirst Sample")
    print("------------")
    print(sample)

    print("\nDataset Info")
    print("------------")
    print(provider.get_info())

    print("\nFirst 3 Samples")
    print("---------------")

    for i, sample in enumerate(provider):
        print(sample)

        if i == 2:
            break


if __name__ == "__main__":
    main()