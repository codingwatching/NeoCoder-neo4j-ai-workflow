import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mcp_neocoder.server import Neo4jWorkflowServer
from mcp_neocoder.tool_registry import registry as tool_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_global_reg")


async def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    logger.info(f"Connecting to Neo4j at {uri}...")

    # Instantiate with connection details
    connection_details = {"url": uri, "username": user, "password": password}
    server = Neo4jWorkflowServer(connection_details=connection_details)

    logger.info("Initializing server (should register ALL tools)...")
    try:
        # We call _initialize_async manually
        await server._initialize_async()

        # Check tools in the registry
        registered_keys = tool_registry._mcp_registered_tools
        logger.info(
            f"Total registered tools tracked in registry: {len(registered_keys)}"
        )

        # We expect tools from different incarnations to be present
        expected_partials = [
            # Core
            "Neo4jWorkflowServer.list_incarnations",
            # RDCF-LV
            "RdcfLvIncarnation.seed_constraints",
            "RdcfLvIncarnation.run_evolution",
            # Knowledge Graph
            "KnowledgeGraphIncarnation.read_graph",
            # Coding
            "CodingIncarnation.get_project",
        ]

        missing = []
        for expected in expected_partials:
            found = False
            for key in registered_keys:
                if expected in key:
                    found = True
                    break
            if not found:
                missing.append(expected)

        if missing:
            logger.error(f"Missing expected tools: {missing}")
            # Print a sample of registered tools
            logger.info("Sample of registered tools:")
            for k in list(registered_keys)[:10]:
                logger.info(f" - {k}")
            sys.exit(1)
        else:
            logger.info("SUCCESS: All expected tools found from multiple incarnations!")
            logger.info("NeoCoder is now a 'bristling bag of tools'!")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
