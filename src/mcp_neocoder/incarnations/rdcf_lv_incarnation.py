"""
RDCF-LV Incarnation (The Ecological Architect) for the NeoCoder framework.

This incarnation integrates the Recursive Domain Construction Framework (RDCF) with
Lotka-Volterra (LV) ecosystem dynamics to evolve robust architectural constraints through
adversarial predation (The Shredder).
"""

import logging
import uuid
from typing import Annotated, Dict, List

import mcp.types as types
import numpy as np
from pydantic import Field

from ..event_loop_manager import safe_neo4j_session
from .base_incarnation import BaseIncarnation

logger = logging.getLogger("mcp_neocoder.incarnations.rdcf_lv")


class RdcfLvIncarnation(BaseIncarnation):
    """
    RDCF-LV Incarnation (The Ecological Architect).

    Merges Recursive Domain Construction Framework (RDCF) with Lotka-Volterra (LV)
    dynamics to create an evolutionary architecture system. Constraints (Prey) are
    subjected to adversarial Shredders (Predators) to test their viability.
    """

    # Define the incarnation name
    name = "rdcf_lv"

    # Metadata for display in the UI
    description = "Ecological Architect: Evolutionary RDCF with Lotka-Volterra Dynamics"
    version = "0.1.0"

    # Schema creation queries
    schema_queries = [
        # Constraints on Constraint Nodes
        "CREATE CONSTRAINT constraint_id IF NOT EXISTS FOR (c:Constraint) REQUIRE c.id IS UNIQUE",
        "CREATE INDEX constraint_status IF NOT EXISTS FOR (c:Constraint) ON (c.status)",
        # Constraints on Shredder Nodes
        "CREATE CONSTRAINT shredder_id IF NOT EXISTS FOR (s:ShredderStrategy) REQUIRE s.id IS UNIQUE",
        # Constraints on Ledger
        "CREATE CONSTRAINT ledger_id IF NOT EXISTS FOR (l:ConstraintLedger) REQUIRE l.id IS UNIQUE",
    ]

    # Hub content
    hub_content = """
# Ecological Architect Hub (RDCF-LV)

Welcome to the **Ecological Architect**. This environment uses biological ecosystem dynamics to evolve software architecture.

## 🧬 Core Concepts

1.  **Constraint Ledger (The Ecosystem)**: The state machine managing architectural invariants.
2.  **Constraints (Prey)**: Architectural rules that want to survive (e.g., "Latency < 50ms", "No circular deps").
3.  **Shredders (Predators)**: Adversarial strategies that hunt weak constraints (e.g., "Load Tester", "Dependency Auditor").
4.  **Lotka-Volterra Dynamics**: Mathematical simulation of the predator-prey relationship to determine constraint viability.

## 🔄 Workflow steps

1.  `seed_constraints()`: Define the initial population of architectural constraints.
2.  `unleash_shredders()`: Define the predatory strategies to test these constraints.
3.  `run_evolution()`: Run the Lotka-Volterra simulation to see which constraints survive.
4.  `finalize_ledger()`: Commit the survivors to the permanent Constraint Ledger.
"""

    async def initialize_schema(self) -> None:
        """Initialize the Neo4j schema for RDCF-LV."""
        try:
            # Run parent initialization
            await super().initialize_schema()

            # Create specific RDCF-LV action templates
            await self._create_rdcf_templates()

            logger.info("RDCF-LV incarnation schema initialized")
        except Exception as e:
            logger.error(f"Error initializing RDCF-LV schema: {e}")
            raise

    async def _create_rdcf_templates(self) -> None:
        """Create Action Templates for RDCF-LV workflows."""
        templates = [
            {
                "keyword": "RDCF_CONSTRUCT",
                "name": "Recursive Domain Construction",
                "description": "Full RDCF-LV construction workflow",
                "steps": """1. Seed Constraints (`seed_constraints`)
2. Define Shredders (`unleash_shredders`)
3. Run Evolution (`run_evolution`)
4. Analyze Survivors
5. Finalize Ledger (`finalize_ledger`)""",
            }
        ]

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                for tmpl in templates:
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
                    MERGE (hub:AiGuidanceHub {id: 'rdcf_lv_hub'})
                    MERGE (hub)-[:PROVIDES_TEMPLATE]->(t)
                    """
                    # Bind loop variables
                    k = tmpl["keyword"]
                    n = tmpl["name"]
                    d = tmpl["description"]
                    s = tmpl["steps"]
                    await session.execute_write(
                        lambda tx, q=query, k=k, n=n, d=d, s=s: tx.run(
                            q, {"keyword": k, "name": n, "description": d, "steps": s}
                        )
                    )
        except Exception as e:
            logger.error(f"Error creating RDCF templates: {e}")

    # -------------------------------------------------------------------------
    # Core Tools
    # -------------------------------------------------------------------------

    async def seed_constraints(
        self,
        project_id: Annotated[str, Field(description="ID of the project")],
        constraints: Annotated[
            List[str],
            Field(description="List of architectural constraints description"),
        ],
        strength: Annotated[
            float, Field(description="Initial strength (population) of constraints")
        ] = 1.0,
    ) -> List[types.TextContent]:
        """
        Seed the ecosystem with initial architectural constraints (Prey).

        Creates a ConstraintLedger and Constraint nodes.
        """
        ledger_id = f"ledger_{project_id}"

        async with safe_neo4j_session(self.driver, self.database) as session:
            # Create Ledger
            await session.execute_write(
                lambda tx: tx.run(
                    """
                MERGE (l:ConstraintLedger {id: $id})
                SET l.project_id = $project_id,
                    l.status = 'SEEDING',
                    l.created_at = datetime()
            """,
                    {"id": ledger_id, "project_id": project_id},
                )
            )

            # Create Constraints
            results = []
            for desc in constraints:
                cid = str(uuid.uuid4())
                await session.execute_write(
                    lambda tx, ledger_id=ledger_id, cid=cid, desc=desc, strength=strength: tx.run(
                        """
                    MATCH (l:ConstraintLedger {id: $ledger_id})
                    CREATE (c:Constraint {
                        id: $id,
                        description: $desc,
                        strength: $strength,
                        status: 'INCUBATING'
                    })
                    MERGE (l)-[:TRACKS]->(c)
                """,
                        {
                            "ledger_id": ledger_id,
                            "id": cid,
                            "desc": desc,
                            "strength": strength,
                        },
                    )
                )
                results.append(f"Seeded: {desc} (ID: {cid})")

        return [
            types.TextContent(
                type="text",
                text=f"Seeded {len(results)} constraints into Ledger {ledger_id}.\n"
                + "\n".join(results),
            )
        ]

    async def unleash_shredders(
        self,
        project_id: Annotated[str, Field(description="ID of the project")],
        shredder_types: Annotated[
            List[str],
            Field(
                description="Types of shredders (e.g., 'SECURITY', 'PERFORMANCE', 'MAINTAINABILITY')"
            ),
        ],
        aggression: Annotated[
            float, Field(description="Initial aggression (predator population) level")
        ] = 0.5,
    ) -> List[types.TextContent]:
        """
        Introduce Shredders (Predators) into the ecosystem.

        These nodes represent adversarial strategies that will 'attack' weak constraints.
        """
        ledger_id = f"ledger_{project_id}"

        async with safe_neo4j_session(self.driver, self.database) as session:
            results = []
            for stype in shredder_types:
                sid = str(uuid.uuid4())
                await session.execute_write(
                    lambda tx, ledger_id=ledger_id, sid=sid, stype=stype, aggression=aggression: tx.run(
                        """
                    MATCH (l:ConstraintLedger {id: $ledger_id})
                    CREATE (s:ShredderStrategy {
                        id: $id,
                        type: $type,
                        aggression: $aggression,
                        status: 'HUNTING'
                    })
                    MERGE (l)-[:DEPLOYED_AGAINST]->(s)
                """,
                        {
                            "ledger_id": ledger_id,
                            "id": sid,
                            "type": stype,
                            "aggression": aggression,
                        },
                    )
                )
                results.append(f"Unleashed: {stype} Shredder (ID: {sid})")

        return [
            types.TextContent(
                type="text",
                text=f"Unleashed {len(results)} shredders against Ledger {ledger_id}.\n"
                + "\n".join(results),
            )
        ]

    async def run_evolution(
        self,
        project_id: Annotated[str, Field(description="ID of the project")],
        steps: Annotated[int, Field(description="Number of simulation steps")] = 100,
    ) -> List[types.TextContent]:
        """
        Run the Lotka-Volterra Lotka-Volterra simulation.

        Evolves the populations of Constraints (Prey) and Shredders (Predators).
        Constraints that drop below a survival threshold are marked as 'CONSUMED'.
        """
        ledger_id = f"ledger_{project_id}"

        async with safe_neo4j_session(self.driver, self.database) as session:
            # Fetch current state
            data = await session.execute_read(
                lambda tx: tx.run(
                    """
                MATCH (l:ConstraintLedger {id: $ledger_id})
                OPTIONAL MATCH (l)-[:TRACKS]->(c:Constraint)
                OPTIONAL MATCH (l)-[:DEPLOYED_AGAINST]->(s:ShredderStrategy)
                RETURN collect(DISTINCT c) as constraints, collect(DISTINCT s) as shredders
            """,
                    {"ledger_id": ledger_id},
                )
            )
            record = await data.single()

            constraints = [c for c in record["constraints"] if c]
            shredders = [s for s in record["shredders"] if s]

            if not constraints or not shredders:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: Ecosystem requires both Constraints and Shredders.",
                    )
                ]

            # --- Lotka-Volterra Simulation Core (Discrete) ---
            # dP/dt = alpha*P - beta*P*S (Prey/Constraints)
            # dS/dt = delta*P*S - gamma*S (Predator/Shredders)

            # Parameters
            alpha = (
                0.1  # Intrinsic growth rate of constraints (validation reinforcement)
            )
            beta = 0.05  # Predation rate (how effective shredders are)
            delta = 0.02  # Shredder growth rate (successful finds reinforce strategy)
            gamma = 0.1  # Shredder decay rate (irrelevant strategies fade)
            dt = 0.1

            # Initial State
            P = np.array([float(c.get("strength", 1.0)) for c in constraints])
            S = np.array([float(s.get("aggression", 0.5)) for s in shredders])

            # Simple interaction matrix (all shredders affect all constraints for MVP)
            # In advanced versions, we'd use embedding similarity to mask interactions

            history: Dict[str, List[List[float]]] = {"P": [], "S": []}

            for _ in range(steps):
                # We perform a mean-field approximation where S is a pool interacting with vector P
                total_S = np.sum(S)
                total_P = np.sum(P)

                # Prey Equation (Constraints)
                # Each constraint grows by intrinsic value, shrinks by TOTAL predation pressure
                dP = (alpha * P) - (beta * P * total_S)

                # Predator Equation (Shredders)
                # Each shredder grows by available PREY biomass, shrinks by decay
                dS = (delta * S * total_P) - (gamma * S)

                P = P + (dP * dt)
                S = S + (dS * dt)

                # Ensure non-negative
                P = np.maximum(P, 0)
                S = np.maximum(S, 0)

                history["P"].append(P.tolist())
                history["S"].append(S.tolist())

            # Analyze Results
            final_P = P
            survival_threshold = 0.5

            survivor_updates = []

            for i, constraint in enumerate(constraints):
                status = "SURVIVED" if final_P[i] > survival_threshold else "CONSUMED"
                cid = constraint["id"]
                final_strength = float(final_P[i])

                # Update DB
                await session.execute_write(
                    lambda tx, cid=cid, st=status, strg=final_strength: tx.run(
                        """
                    MATCH (c:Constraint {id: $cid})
                    SET c.status = $st, c.strength = $strg
                """,
                        {"cid": cid, "st": st, "strg": strg},
                    )
                )

                survivor_updates.append(
                    f"- {constraint['description']}: {status} (Strength: {final_strength:.2f})"
                )

        return [
            types.TextContent(
                type="text",
                text=f"""
# Evolution Complete
Simulation ran for {steps} steps.

## Results
{chr(10).join(survivor_updates)}

## Ecosystem Metrics
- Final Biomass (Constraints): {np.sum(P):.2f}
- Final Predation Pressure: {np.sum(S):.2f}
""",
            )
        ]

    async def finalize_ledger(
        self,
        project_id: Annotated[str, Field(description="ID of the project")],
    ) -> List[types.TextContent]:
        """
        Finalize the Constraint Ledger.

        Locks the survivors as permanent architectural laws for the project.
        """
        ledger_id = f"ledger_{project_id}"

        async with safe_neo4j_session(self.driver, self.database) as session:
            result = await session.execute_write(
                lambda tx: tx.run(
                    """
                MATCH (l:ConstraintLedger {id: $ledger_id})
                SET l.status = 'ACTIVE', l.finalized_at = datetime()
                WITH l
                MATCH (l)-[:TRACKS]->(c:Constraint {status: 'SURVIVED'})
                RETURN count(c) as survivors
            """,
                    {"ledger_id": ledger_id},
                )
            )
            record = await result.single()
            count = record["survivors"] if record else 0

        return [
            types.TextContent(
                type="text",
                text=f"Ledger {ledger_id} finalized. {count} constraints are now active laws.",
            )
        ]
