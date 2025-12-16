"""
Coding Incarnation for the NeoCoder framework.

This incarnation provides the original NeoCoder functionality for AI-assisted coding workflows,
including action templates, project management, workflow tracking, and best practices guidance.
"""

import logging
from typing import List

import mcp.types as types

from ..action_templates import ActionTemplateMixin
from ..event_loop_manager import safe_neo4j_session
from .base_incarnation import BaseIncarnation

logger = logging.getLogger("mcp_neocoder.incarnations.coding")


class CodingIncarnation(BaseIncarnation, ActionTemplateMixin):
    """
    Coding Incarnation for the NeoCoder framework.

    This is the original NeoCoder incarnation that provides structured coding workflows,
    action templates (FIX, REFACTOR, DEPLOY, etc.), project management, and workflow tracking.
    """

    # Define the incarnation name
    name = "coding"

    # Metadata for display in the UI
    description = "Original NeoCoder for AI-assisted coding workflows"
    version = "1.0.0"

    # Only register coding-specific tools here (if any in the future)
    _tool_methods = []

    # Only include coding-specific schema queries (project, workflow, file, directory)
    schema_queries = [
        # Drop any existing conflicting indexes before creating constraints
        "DROP INDEX file_path IF EXISTS",
        # Project constraints
        "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
        # Workflow execution constraints
        "CREATE CONSTRAINT workflow_id IF NOT EXISTS FOR (w:WorkflowExecution) REQUIRE w.id IS UNIQUE",
        # File and directory constraints (project-scoped to allow multiple projects)
        "CREATE CONSTRAINT unique_file_path IF NOT EXISTS FOR (f:File) REQUIRE (f.project_id, f.path) IS UNIQUE",
        "CREATE CONSTRAINT unique_dir_path IF NOT EXISTS FOR (d:Directory) REQUIRE (d.project_id, d.path) IS UNIQUE",
        # Indexes for efficient querying
        "CREATE INDEX project_name IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX workflow_timestamp IF NOT EXISTS FOR (w:WorkflowExecution) ON (w.timestamp)",
        "CREATE INDEX file_name IF NOT EXISTS FOR (f:File) ON (f.name)",
        "CREATE INDEX dir_name IF NOT EXISTS FOR (d:Directory) ON (d.name)",
    ]

    # Hub content - guidance for coding workflows
    hub_content = """
# NeoCoder Coding Incarnation

You are now in the **Coding Incarnation**. This environment is optimized for software development, debugging, and refactoring tasks.

## 📍 Where You Are
This Hub is your central navigation point. It is connected to specific **Action Templates** and **Projects** in the knowledge graph.

## 🔍 How to Find Information
*   **Action Templates**: Look at the "Available Action Templates" list below. These are structured workflows for common tasks like `FIX` or `REFACTOR`.
    *   To see the *steps* for a workflow: `get_action_template(keyword="...")`
*   **Projects**: To see managed repositories: `list_projects()`
*   **History**: To see what has been done before: `get_workflow_history()`

## 🛠️ Key Commands
*   `get_action_template(keyword=...)` - Retrieve the specific instructions for a task.
*   `log_workflow_execution(...)` - Record your work (Required: tests must pass!).
*   `switch_incarnation(...)` - Change to a different mode (e.g., Research, Data Analysis).

The system will now list the available workflows found in the graph:
"""

    async def initialize_schema(self) -> None:
        """Initialize the Neo4j schema for the Coding incarnation."""
        try:
            # First run the parent class initialization
            await super().initialize_schema()

            # Then create the coding-specific action templates
            await self._create_action_templates()

            # Create sample projects if needed
            await self._create_sample_projects()

            # Create best practices guide
            await self._create_best_practices()

            logger.info("Coding incarnation schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing coding schema: {e}")
            raise

    async def verify_package(
        self,
        package_name: str,
        ecosystem: str = "pypi",
    ) -> List[types.TextContent]:
        """
        Verify a software package against known repositories to prevent 'slop squatting'.

        This tool simulates checking public registries (like PyPI or npm) to ensure a package
        exists, is maintained, and is not a malicious typosquat.

        Args:
            package_name: The name of the package to verify (e.g., 'pandas', 'requests').
            ecosystem: The package ecosystem ('pypi', 'npm'). Default is 'pypi'.
        """
        # Simulation logic (since we lack direct internet access in this specific environment context)
        # In a production environment, this would call PyPI/npm APIs or use `pip-audit`.

        known_safe = {
            "pandas",
            "numpy",
            "requests",
            "flask",
            "fastapi",
            "django",
            "scikit-learn",
            "matplotlib",
            "pytest",
            "black",
            "mypy",
            "ruff",
        }

        status = "UNKNOWN"
        risk_level = "HIGH"
        message = "Package not found in known-safe list. Verify manually."

        if package_name.lower() in known_safe:
            status = "VERIFIED"
            risk_level = "LOW"
            message = "Package is a known safe dependency."

        result = f"""
## Dependency Verification Report
**Package:** `{package_name}`
**Ecosystem:** {ecosystem}
**Status:** {status}
**Risk Level:** {risk_level}

{message}

> [!WARNING]
> This is a simulated check. In a live environment, use `pip-audit` or query PyPI directly to prevent supply chain attacks.
"""
        return [types.TextContent(type="text", text=result)]

    async def _create_action_templates(self) -> None:
        """Create the standard action templates for coding workflows."""
        templates = [
            {
                "keyword": "FIX",
                "name": "Bug Fix Workflow",
                "description": "Structured approach to fixing bugs with mandatory testing",
                "steps": """1. Review the bug report and understand the issue
2. Locate the relevant code files
3. Analyze the root cause
4. Implement the fix
5. Run ALL relevant tests (mandatory!)
6. Update documentation if needed
7. Create or update tests for the bug
8. Verify the fix resolves the issue""",
            },
            {
                "keyword": "REFACTOR",
                "name": "Code Refactoring Workflow",
                "description": "Safe refactoring with regression testing",
                "steps": """1. Identify the code to refactor
2. Understand current functionality
3. Run existing tests to establish baseline
4. Plan the refactoring approach
5. Implement refactoring incrementally
6. Run tests after each change
7. Ensure no functionality is broken
8. Update documentation""",
            },
            {
                "keyword": "DEPLOY",
                "name": "Deployment Workflow",
                "description": "Safe deployment process with verification",
                "steps": """1. Review all changes to be deployed
2. Run full test suite
3. Check deployment checklist
4. Prepare deployment notes
5. Execute deployment process
6. Verify deployment success
7. Monitor for issues
8. Document deployment""",
            },
            {
                "keyword": "FEATURE",
                "name": "Feature Implementation Workflow",
                "description": "Adding new features with proper testing",
                "steps": """1. Review feature requirements
2. Design implementation approach
3. Create feature branch
4. Implement feature incrementally
5. Write tests for new functionality
6. Run all tests
7. Update documentation
8. Prepare for code review""",
            },
            {
                "keyword": "TOOL_ADD",
                "name": "Add Tool to NeoCoder",
                "description": "Process for adding new tools to the MCP server",
                "steps": """1. Define tool purpose and parameters
2. Add tool method to appropriate module
3. Include proper type hints and docstring
4. Register tool in server initialization
5. Test tool functionality
6. Update documentation
7. Add to tool suggestions
8. Create usage examples""",
            },
            {
                "keyword": "CODE_ANALYZE",
                "name": "Code Analysis Workflow",
                "description": "Analyzing code structure and quality",
                "steps": """1. Select code to analyze
2. Run AST/ASG analysis tools
3. Review code metrics
4. Identify code smells
5. Document findings
6. Suggest improvements
7. Create refactoring plan if needed
8. Update project documentation""",
            },
            {
                "keyword": "SHRED",
                "name": "Design Doc Shredder Review",
                "description": "Adversarial design review using the Principal Engineer Checklist",
                "steps": """1. **Ambiguity Scan**: Highlight vague verbs (e.g., 'manage', 'process'). Demand mechanical definitions.
2. **Dependency Audit**: Extract every library name. Use `verify_package` to confirm existence. Reject non-verified packages.
3. **Failure Analysis (What If?)**:
   - What if invalid input (1GB file)?
   - What if network partition?
   - What if database locks?
4. **Observability Check**: Ensure logging (JSON), metrics (latency/errors), and health checks are defined.
5. **Security & Data**: Verify authentication protocols and data encryption.
6. **Go/No-Go Decision**: Only approve if all 'Traps' are mitigated.""",
            },
        ]

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                for template in templates:
                    query = """
                    MERGE (t:ActionTemplate {keyword: $keyword})
                    SET t.name = $name,
                        t.description = $description,
                        t.steps = $steps,
                        t.isCurrent = true,
                        t.version = 1,
                        t.created = datetime(),
                        t.updated = datetime()

                    WITH t
                    MERGE (hub:AiGuidanceHub {id: 'coding_hub'})
                    MERGE (hub)-[:PROVIDES_TEMPLATE]->(t)
                    """
                    # Bind loop variables for closure
                    q = query
                    t = template
                    await session.execute_write(lambda tx, q=q, t=t: tx.run(q, t))
                    logger.info(f"Created action template: {template['keyword']}")
        except Exception as e:
            logger.error(f"Error creating action templates: {e}")
            raise

    async def _create_sample_projects(self) -> None:
        """Create sample projects if none exist."""
        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                # Check if any projects exist
                result = await session.run("MATCH (p:Project) RETURN count(p) as count")
                data = await result.single()

                if data and data["count"] == 0:
                    # Create a sample project
                    query = """
                    CREATE (p:Project {
                        id: 'neocoder_project',
                        name: 'NeoCoder System',
                        description: 'The NeoCoder MCP server system',
                        readme: 'This is the NeoCoder system for AI-assisted coding workflows.',
                        created: datetime(),
                        updated: datetime()
                    })
                    """
                    await session.execute_write(lambda tx: tx.run(query))
                    logger.info("Created sample project")
        except Exception as e:
            logger.error(f"Error creating sample projects: {e}")

    async def _create_best_practices(self) -> None:
        """Create the best practices guide."""
        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                query = """
                MERGE (bp:BestPracticesGuide {id: 'main'})
                SET bp.content = $content,
                    bp.updated = datetime()
                """

                content = """# NeoCoder Principal Engineer Checklist (The Shredder)

## 1. Ambiguity & Scope
*   **The "How" Test**: Does every verb (process, handle) have a mechanism?
*   **The Boundary Test**: What is *not* being built?
*   **The NFR Test**: Are throughput/latency defined with numbers?

## 2. Supply Chain Security
*   **The Existence Test**: Do all packages exist? (Use `verify_package`).
*   **The Freshness Test**: Are packages maintained?
*   **The Weight Test**: Are we using a sledgehammer (Django) for a nut (script)?

## 3. Failure Modes (What If?)
*   **The Resource Test**: What happens if a 10GB file is uploaded? (DoS).
*   **The Network Test**: What happens if the DB connection times out?
*   **The State Test**: What happens if the process crashes mid-transaction?

## 4. Observability
*   **The Log Test**: Are logs structured (JSON)? No `print()` allowed.
*   **The Metric Test**: Are Success Rate and Latency measured?
*   **The Probe Test**: Is there a `/health` endpoint?

## 5. Standard Code Quality
*   Always write tests before marking work as complete.
*   Keep functions focused and small.
*   Document complex logic.
"""

                await session.execute_write(
                    lambda tx: tx.run(query, {"content": content})
                )
                logger.info("Created best practices guide")
        except Exception as e:
            logger.error(f"Error creating best practices: {e}")


# End of coding incarnation
