from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

def main():
    # 1. Automatically create all production directories defined in config
    ProjectConfig.initialize()

    # 2. Initialize the standardized project logger
    logger = BridgeLogger.get_logger("FrameworkTest")

    logger.info("✅ Bridge framework initialized successfully.")
    logger.info("✅ Log files are being routed correctly.")

    print("\nProject Root Identified:")
    print("-------------------------")
    print(ProjectConfig.PROJECT_ROOT)

    print("\nNext Action: Check datasets/logs/bridgedeux.log to verify storage.")

if __name__ == "__main__":
    main()