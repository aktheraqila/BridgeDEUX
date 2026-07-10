from pprint import pprint

from bridge.metadata import MetadataCollector


def main():

    pprint(MetadataCollector.as_dict())


if __name__ == "__main__":
    main()