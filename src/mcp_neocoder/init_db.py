"""
Initialization Script for NeoCoder Polymorphic Framework

This script initializes the Neo4j database with the necessary schemas for each incarnation.
It creates the base nodes and relationships needed for the system to function properly.
"""

import logging
import os
import sys
from typing import List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from .event_loop_manager import safe_neo4j_session

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_neocoder_init")

# No hardcoded types - they're discovered dynamically


# Dynamically discover available incarnation names
def discover_incarnation_types() -> List[str]:
    """Discover available incarnation names from the filesystem."""
    import os

    incarnation_names = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    incarnations_dir = os.path.join(current_dir, "incarnations")

    if not os.path.exists(incarnations_dir):
        logger.warning(f"Incarnations directory not found: {incarnations_dir}")
        return []  # Return empty list if directory doesn't exist

    # Simple discovery approach: scan for *_incarnation.py files
    for entry in os.listdir(incarnations_dir):
        if not entry.endswith("_incarnation.py") or entry.startswith("__"):
            continue

        # Skip base incarnation
        if entry == "base_incarnation.py":
            continue

        # Extract name from filename (e.g., "data_analysis" from "data_analysis_incarnation.py")
        incarnation_name = entry[:-14]  # Remove "_incarnation.py" (14 chars)
        if incarnation_name:
            # Make sure there's no trailing underscore
            if incarnation_name.endswith("_"):
                incarnation_name = incarnation_name[:-1]
            incarnation_names.append(incarnation_name)
            logger.info(f"Discovered incarnation name: {incarnation_name} from {entry}")

    return incarnation_names


async def init_neo4j_connection(uri: str, user: str, password: str) -> AsyncDriver:
    """Initialize Neo4j connection."""
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        # Test the connection
        async with safe_neo4j_session(driver, "neo4j") as session:
            await session.run("RETURN 1")
        logger.info("Successfully connected to Neo4j")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)


async def init_base_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Initialize the base schema that is common to all incarnations."""
    logger.info("Initializing base schema...")

    # Base constraints and indexes
    base_schema_queries = [
        # Guidance Hub
        "CREATE CONSTRAINT hub_id IF NOT EXISTS FOR (hub:AiGuidanceHub) REQUIRE hub.id IS UNIQUE",
        # Tool proposals
        "CREATE CONSTRAINT proposal_id IF NOT EXISTS FOR (p:ToolProposal) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT request_id IF NOT EXISTS FOR (r:ToolRequest) REQUIRE r.id IS UNIQUE",
        # Cypher snippets
        "CREATE CONSTRAINT snippet_id IF NOT EXISTS FOR (s:CypherSnippet) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
        # Core Workflow Entities
        "CREATE CONSTRAINT unique_action_template_current IF NOT EXISTS FOR (t:ActionTemplate) REQUIRE (t.keyword, t.isCurrent) IS UNIQUE",
        "CREATE CONSTRAINT unique_project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.projectId IS UNIQUE",
        "CREATE CONSTRAINT unique_workflow_execution_id IF NOT EXISTS FOR (w:WorkflowExecution) REQUIRE w.id IS UNIQUE",
        # Indexes
        "CREATE INDEX hub_type IF NOT EXISTS FOR (hub:AiGuidanceHub) ON (hub.type)",
        "CREATE INDEX proposal_status IF NOT EXISTS FOR (p:ToolProposal) ON (p.status)",
        "CREATE INDEX request_status IF NOT EXISTS FOR (r:ToolRequest) ON (r.status)",
        "CREATE INDEX snippet_name IF NOT EXISTS FOR (s:CypherSnippet) ON (s.name)",
        "CREATE INDEX action_template_keyword IF NOT EXISTS FOR (t:ActionTemplate) ON (t.keyword)",
        "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query in base_schema_queries:
                await session.run(query)
        logger.info("Base schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing base schema: {e}")
        raise


async def init_research_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Initialize the schema for the Research Orchestration Platform incarnation."""
    logger.info("Initializing research schema...")

    research_schema_queries = [
        # Constraints
        "CREATE CONSTRAINT research_hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
        "CREATE CONSTRAINT research_experiment_id IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT research_protocol_id IF NOT EXISTS FOR (p:Protocol) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT research_observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT research_run_id IF NOT EXISTS FOR (r:Run) REQUIRE r.id IS UNIQUE",
        # Indexes
        "CREATE INDEX research_hypothesis_status IF NOT EXISTS FOR (h:Hypothesis) ON (h.status)",
        "CREATE INDEX research_experiment_status IF NOT EXISTS FOR (e:Experiment) ON (e.status)",
        "CREATE INDEX research_protocol_name IF NOT EXISTS FOR (p:Protocol) ON (p.name)",
        # Create research hub
        """
        MERGE (hub:AiGuidanceHub {id: 'research_hub'})
        ON CREATE SET
            hub.description = 'Research Orchestration Platform - A system for managing scientific workflows, hypotheses, experiments, and observations.'
        RETURN hub
        """,
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query in research_schema_queries:
                await session.run(query)
        logger.info("Research schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing research schema: {e}")
        raise


async def init_decision_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Initialize the schema for the Decision Support System incarnation."""
    logger.info("Initializing decision support schema...")

    decision_schema_queries = [
        # Constraints
        "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT alternative_id IF NOT EXISTS FOR (a:Alternative) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE",
        # Indexes
        "CREATE INDEX decision_status IF NOT EXISTS FOR (d:Decision) ON (d.status)",
        "CREATE INDEX alternative_name IF NOT EXISTS FOR (a:Alternative) ON (a.name)",
        # Create decision hub
        """
        MERGE (hub:AiGuidanceHub {id: 'decision_hub'})
        ON CREATE SET
            hub.description = 'Decision Support System - A system for tracking decisions, alternatives, metrics, and evidence to support transparent, data-driven decision-making.'
        RETURN hub
        """,
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query in decision_schema_queries:
                await session.run(query)
        logger.info("Decision schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing decision schema: {e}")
        raise


async def init_learning_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Initialize the schema for the Continuous Learning Environment incarnation."""
    logger.info("Initializing learning schema...")

    learning_schema_queries = [
        # Constraints
        "CREATE CONSTRAINT learning_user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT learning_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT learning_problem_id IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT learning_attempt_id IF NOT EXISTS FOR (a:Attempt) REQUIRE a.id IS UNIQUE",
        # Indexes
        "CREATE INDEX learning_user_name IF NOT EXISTS FOR (u:User) ON (u.name)",
        "CREATE INDEX learning_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        "CREATE INDEX learning_problem_difficulty IF NOT EXISTS FOR (p:Problem) ON (p.difficulty)",
        # Create learning hub
        """
        MERGE (hub:AiGuidanceHub {id: 'learning_hub'})
        ON CREATE SET
            hub.description = 'Continuous Learning Environment - A system for adaptive learning and personalized content delivery based on knowledge spaces and mastery tracking.'
        RETURN hub
        """,
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query in learning_schema_queries:
                await session.run(query)
        logger.info("Learning schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing learning schema: {e}")
        raise


async def init_simulation_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Initialize the schema for the Complex System Simulation incarnation."""
    logger.info("Initializing simulation schema...")

    simulation_schema_queries = [
        # Constraints
        "CREATE CONSTRAINT simulation_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT simulation_model_id IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT simulation_run_id IF NOT EXISTS FOR (r:SimulationRun) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT simulation_state_id IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE",
        # Indexes
        "CREATE INDEX simulation_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX simulation_model_name IF NOT EXISTS FOR (m:Model) ON (m.name)",
        "CREATE INDEX simulation_run_timestamp IF NOT EXISTS FOR (r:SimulationRun) ON (r.timestamp)",
        # Create simulation hub
        """
        MERGE (hub:AiGuidanceHub {id: 'simulation_hub'})
        ON CREATE SET
            hub.description = 'Complex System Simulation - A system for modeling and simulating complex systems with multiple interacting components and emergent behavior.'
        RETURN hub
        """,
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query in simulation_schema_queries:
                await session.run(query)
        logger.info("Simulation schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing simulation schema: {e}")
        raise


async def create_main_guidance_hub(
    driver: AsyncDriver, database: str = "neo4j"
) -> None:
    """Create the main guidance hub that lists all available incarnations."""
    logger.info("Creating main guidance hub...")

    main_hub_description = """
# NeoCoder Polymorphic Framework

Welcome to the NeoCoder Polymorphic Framework. This system can transform between multiple incarnations to support different use cases:

## Default Available Incarnation Templates
- **base_incarnation**: The base NeoCoder incarnation, providing core functionality.
- **coding_incarnation**: The default coding incarnation, focused on AI-assisted coding tasks.
- **data_analysis_incarnation**: Data Analysis Incarnation
- **decision_incarnation**: Decision Support System
- **knowledge_graph_incarnation**: Knowledge Graph Incarnation
- **research_incarnation**: Research Orchestration Platform

## Getting Started

- Use `list_incarnations()` to see all available incarnations
- Use `switch_incarnation(incarnation_type="research")` to activate a specific incarnation
- Once activated, use `get_guidance_hub()` again to see incarnation-specific guidance

Each incarnation provides its own set of specialized tools while maintaining core Neo4j integration.
    """

    query = """
    MERGE (hub:AiGuidanceHub {id: 'main_hub'})
    ON CREATE SET hub.description = $description
    RETURN hub
    """

    try:
        async with safe_neo4j_session(driver, database) as session:
            await session.run(query, {"description": main_hub_description})
        logger.info("Main guidance hub created successfully")

        # Create core guides linked to main hub
        await create_core_guides(driver, database)

    except Exception as e:
        logger.error(f"Error creating main guidance hub: {e}")
        raise


async def create_core_guides(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Create the core guide nodes and link them to the main hub."""
    logger.info("Creating core guide nodes...")

    # Best Practices Guide
    bp_content = """Core Coding & System Practices:
- **Efficiency First:** Prefer editing existing code over complete rewrites where feasible. Avoid temporary patch files.
- **Meaningful Naming:** Do not name functions, variables, or files 'temp', 'fixed', 'patch'. Use descriptive names reflecting purpose.
- **README is Key:** ALWAYS review the project's README before starting work. Find it via the :Project node.
- **Test Rigorously:** Before logging completion, ALL relevant tests must pass. If tests fail, revisit the code, do not log success.
- **Update After Success:** ONLY AFTER successful testing, update the Neo4j project tree AND the project's README with changes made.
- **Risk Assessment:** Always evaluate the potential impact of changes and document any areas that need monitoring.
- **Metrics Collection:** Track completion time and success rates to improve future estimation accuracy."""

    # Templating Guide
    tg_content = """How to Create/Edit ActionTemplates:
-   Nodes are `:ActionTemplate {keyword: STRING, version: STRING, isCurrent: BOOLEAN, description: STRING, steps: STRING}`.
-   `keyword`: Short, unique verb (e.g., 'DEPLOY', 'TEST_COMPONENT'). Used for lookup.
-   `version`: Semantic version (e.g., '1.0', '1.1').
-   `isCurrent`: Only one template per keyword should be `true`. Use transactions to update.
-   `description`: Brief explanation of the template's purpose.
-   `complexity`: Estimation of task complexity (e.g., 'LOW', 'MEDIUM', 'HIGH').
-   `estimatedEffort`: Estimated time in minutes to complete the task.
-   `steps`: Detailed, multi-line string with numbered steps. Use Markdown for formatting. MUST include critical checkpoints like 'Test Verification' and 'Log Successful Execution'.

When updating a template:
1. Create new version with incremented version number
2. Set isCurrent = true on new version
3. Set isCurrent = false on old version
4. Document changes in a :Feedback node"""

    # System Usage Guide
    sg_content = """Neo4j System Overview:
-   `:AiGuidanceHub`: Your starting point.
-   `:Project`: Represents a codebase. Has `projectId`, `name`, `readmeContent`/`readmeUrl`.
-   `:ActionTemplate`: Contains steps for a keyword task. Query by `{keyword: $kw, isCurrent: true}`.
-   `:File`, `:Directory`: Represent code structure within a project. Linked via `CONTAINS`, have `path`, `project_id`.
-   `:WorkflowExecution`: Logs a completed action. Links via `APPLIED_TO_PROJECT` to `:Project`, `MODIFIED` to `:File`/`:Directory`, `USED_TEMPLATE` to `:ActionTemplate`.
-   `:Feedback`: Stores feedback on template effectiveness. Links to templates via `REGARDING`.
-   `:BestPracticesGuide`, `:TemplatingGuide`, `:SystemUsageGuide`: Linked from `:AiGuidanceHub` for help.
-   Always use parameters ($projectId, $keyword) in queries for safety and efficiency.

Common Metrics to Track:
-   Success rate per template
-   Average execution time per template
-   Number of test failures before success
-   Frequency of template usage
-   Most commonly modified files"""

    queries = [
        # Best Practices
        (
            """
        MERGE (hub:AiGuidanceHub {id: 'main_hub'})
        MERGE (bp:BestPracticesGuide {id: 'core_practices'})
        ON CREATE SET bp.content = $content
        MERGE (hub)-[:LINKS_TO]->(bp)
        """,
            {"content": bp_content},
        ),
        # Templating Guide
        (
            """
        MERGE (hub:AiGuidanceHub {id: 'main_hub'})
        MERGE (tg:TemplatingGuide {id: 'template_guide'})
        ON CREATE SET tg.content = $content
        MERGE (hub)-[:LINKS_TO]->(tg)
        """,
            {"content": tg_content},
        ),
        # System Usage Guide
        (
            """
        MERGE (hub:AiGuidanceHub {id: 'main_hub'})
        MERGE (sg:SystemUsageGuide {id: 'system_guide'})
        ON CREATE SET sg.content = $content
        MERGE (hub)-[:LINKS_TO]->(sg)
        """,
            {"content": sg_content},
        ),
    ]

    try:
        async with safe_neo4j_session(driver, database) as session:
            for query, params in queries:
                await session.run(query, params)
        logger.info("Core guide nodes created successfully")
    except Exception as e:
        logger.error(f"Error creating core guide nodes: {e}")
        # Non-critical, do not raise


async def load_templates_from_directory(driver: AsyncDriver, template_dir: str) -> None:
    """Load template files from directory and execute them."""
    import os

    try:
        if not os.path.exists(template_dir):
            logger.warning(f"Template directory not found: {template_dir}")
            return

        template_files = [f for f in os.listdir(template_dir) if f.endswith(".cypher")]
        if not template_files:
            logger.info(f"No .cypher templates found in {template_dir}")
            return

        logger.info(f"Found {len(template_files)} template files in {template_dir}")

        async with safe_neo4j_session(driver, "neo4j") as session:
            for template_file in template_files:
                file_path = os.path.join(template_dir, template_file)
                try:
                    with open(file_path, "r") as f:
                        template_query = f.read()

                    if template_query.strip():
                        # Use split by semicolon if we suspect multiple statements,
                        # but for now let's try running the whole block.
                        # Some cypher files might contain multiple statements separated by ; which need separate runs.
                        # Simple split for now:
                        statements = [
                            s.strip() for s in template_query.split(";") if s.strip()
                        ]
                        for stmt in statements:
                            # Skip comments-only
                            if not stmt.startswith("//"):
                                await session.run(stmt)

                        logger.info(f"Executed template: {template_file}")
                except Exception as e:
                    logger.error(f"Error executing template {template_file}: {e}")

    except Exception as e:
        logger.error(f"Error loading templates: {e}")


async def create_dynamic_links_between_hubs(
    driver: AsyncDriver,
    database: str = "neo4j",
    incarnations: Optional[List[str]] = None,
) -> None:
    """Create relationships between the main hub and all incarnation-specific hubs."""
    logger.info("Creating dynamic links between hubs...")

    # Get all AiGuidanceHub nodes
    query = """
    MATCH (hub:AiGuidanceHub)
    WHERE hub.id <> 'main_hub' AND hub.id ENDS WITH '_hub'
    RETURN hub.id AS id
    """

    try:
        async with safe_neo4j_session(driver, database) as session:
            # Get all hub IDs using to_eager_result() instead
            result = await session.run(query)
            eager_result = await result.to_eager_result()

            # Create links for each hub
            main_hub_query = """
            MATCH (main:AiGuidanceHub {id: 'main_hub'})
            MATCH (inc:AiGuidanceHub {id: $hub_id})
            MERGE (main)-[:HAS_INCARNATION {type: $inc_type}]->(inc)
            """

            for record in eager_result.records:
                hub_id = record.get("id")
                if hub_id:
                    # Extract incarnation type from hub ID (e.g., research_hub -> research)
                    inc_type = hub_id.replace("_hub", "")

                    # Create link
                    await session.run(
                        main_hub_query, {"hub_id": hub_id, "inc_type": inc_type}
                    )
                    logger.info(f"Created link for incarnation: {inc_type}")

        logger.info("Hub links created successfully")
    except Exception as e:
        logger.error(f"Error creating hub links: {e}")
        raise


async def init_db(
    incarnations: Optional[List[str]] = None, template_dir: Optional[str] = None
) -> bool:
    """Initialize the database with the schemas for the specified incarnations."""
    # Get Neo4j connection info from environment variables
    neo4j_uri = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")

    # Discover available incarnation types
    available_incarnations = discover_incarnation_types()
    logger.info(f"Discovered incarnation types: {available_incarnations}")

    # If no incarnations specified, initialize all discovered
    incarnations = incarnations or available_incarnations

    # Connect to Neo4j
    try:
        driver = await init_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        # Return instead of sys.exit to allow recovery
        return False

    success = False
    try:
        # Initialize base schema
        await init_base_schema(driver, neo4j_database)

        # Initialize incarnation-specific schemas
        if "research_orchestration" in incarnations or "research" in incarnations:
            await init_research_schema(driver, neo4j_database)

        if "decision_support" in incarnations or "decision" in incarnations:
            await init_decision_schema(driver, neo4j_database)

        if "continuous_learning" in incarnations or "learning" in incarnations:
            await init_learning_schema(driver, neo4j_database)

        if "complex_system" in incarnations or "simulation" in incarnations:
            await init_simulation_schema(driver, neo4j_database)

        # Create incarnation-specific schemas for other discovered incarnations
        for inc_type in incarnations:
            # Skip already handled incarnations
            if inc_type in [
                "research_orchestration",
                "research",
                "decision_support",
                "decision",
                "continuous_learning",
                "learning",
                "complex_system",
                "simulation",
            ]:
                continue

            logger.info(f"Creating basic schema for incarnation: {inc_type}")
            hub_id = f"{inc_type}_hub"

            # Create a more informative hub description based on the incarnation type
            if inc_type == "knowledge_graph":
                hub_description = "Knowledge Graph Management - Create and analyze semantic knowledge graphs with entities, observations, and relationships."
            elif inc_type == "code_analysis":
                hub_description = "Code Analysis - Parse, analyze, and document code structure, patterns, and metrics."
            elif inc_type == "data_analysis":
                hub_description = "Data Analysis - Analyze datasets, create visualizations, and extract insights from data."
            else:
                # Generic description for any other incarnation
                hub_description = f"This is the {inc_type.replace('_', ' ').title()} incarnation of the NeoCoder framework."

            try:
                # Use parameterized query for safety and to avoid quoting issues
                hub_query = """
                MERGE (hub:AiGuidanceHub {id: $hub_id})
                ON CREATE SET hub.description = $hub_description
                RETURN hub
                """

                async with safe_neo4j_session(driver, neo4j_database) as session:
                    await session.run(
                        hub_query,
                        {"hub_id": hub_id, "hub_description": hub_description},
                    )
                    logger.info(f"Created hub for {inc_type}")

            except Exception as e:
                logger.error(f"Error creating hub for {inc_type}: {e}")

        # Create main guidance hub
        await create_main_guidance_hub(driver, neo4j_database)

        # Create links between hubs for all discovered incarnations
        await create_dynamic_links_between_hubs(driver, neo4j_database, incarnations)

        # Load templates if directory provided
        if template_dir:
            await load_templates_from_directory(driver, template_dir)

        logger.info("Database initialization complete!")
        success = True
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        import traceback

        logger.error(f"Initialization stack trace: {traceback.format_exc()}")
        success = False

    if "driver" in locals():
        await driver.close()

    return success


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the database initialization."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Initialize NeoCoder Database")
    parser.add_argument(
        "--incarnations", nargs="*", help="Specific incarnations to initialize"
    )
    parser.add_argument(
        "--template-dir", help="Directory containing .cypher templates to load"
    )
    args = parser.parse_args(argv)

    # Robust event loop handling for asyncio (fixes 'Future attached to a different loop' errors)
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already running (e.g., in Jupyter, VS Code, or nested async), create a task and wait for it
            task = loop.create_task(init_db(args.incarnations, args.template_dir))

            # If in a REPL, we can't block, so use add_done_callback to log result
            def _log_result(fut: asyncio.Future) -> None:
                try:
                    result = fut.result()
                    logger.info(f"Database initialization result: {result}")
                except Exception as e:
                    logger.error(f"Database initialization failed: {e}")

            task.add_done_callback(_log_result)
            # If possible, run until complete (only if not in a REPL)
            if hasattr(loop, "run_until_complete"):
                loop.run_until_complete(task)
        else:
            asyncio.run(init_db(args.incarnations, args.template_dir))
    except Exception as e:
        logger.error(f"Error running database initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
