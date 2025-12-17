import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from neo4j import AsyncGraphDatabase

from mcp_neocoder.incarnations.rdcf_lv_incarnation import RdcfLvIncarnation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_rdcf_lv")


async def main() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    logger.info(f"Connecting to Neo4j at {uri}")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    try:
        # Instantiate Incarnation
        incarnation = RdcfLvIncarnation(driver=driver, database="neo4j")

        # 1. Initialize Schema (Idempotent)
        logger.info("--- Step 1: Initialize Schema ---")
        await incarnation.initialize_schema()

        project_id = "test_rdcf_project"

        # 2. Seed Constraints
        logger.info("--- Step 2: Seed Constraints ---")
        res = await incarnation.seed_constraints(
            project_id=project_id,
            constraints=[
                "Response time under 50ms",
                "No circular dependencies",
                "99.9% uptime",
                "Strict typing compliance",
            ],
            strength=1.0,
        )
        print(res[0].text)

        # 3. Unleash Shredders
        logger.info("--- Step 3: Unleash Shredders ---")
        res = await incarnation.unleash_shredders(
            project_id=project_id,
            shredder_types=["LOAD_TESTER", "CHAOS_MONKEY", "SECURITY_AUDIT"],
            aggression=0.6,
        )
        print(res[0].text)

        # 4. Run Evolution
        logger.info("--- Step 4: Run Evolution ---")
        res = await incarnation.run_evolution(project_id=project_id, steps=50)
        print(res[0].text)

        # 5. Finalize Ledger
        logger.info("--- Step 5: Finalize Ledger ---")
        res = await incarnation.finalize_ledger(project_id=project_id)
        print(res[0].text)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await driver.close()
        logger.info("Driver closed")


if __name__ == "__main__":
    asyncio.run(main())
