
import logging
import os
import sys

# Create a reproduction script to test incarnation discovery
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("test_discovery")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from mcp_neocoder.incarnation_registry import registry

    logger.info("Starting discovery...")
    registry.discover()

    logger.info(f"Discovered incarnations: {list(registry.incarnations.keys())}")

    if not registry.incarnations:
        logger.error("No incarnations found!")
        # Try direct discovery
        logger.info("Trying direct discovery...")
        identifiers = registry.discover_incarnation_identifiers()
        logger.info(f"Identifiers found: {identifiers}")
    else:
        logger.info("Success!")

except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
