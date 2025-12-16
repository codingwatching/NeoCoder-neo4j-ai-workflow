import asyncio
import logging
import os
import sys
import uuid

from neo4j import AsyncGraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("seed_cypher_snippets")

# Define the snippets data
SNIPPETS = [
    # Node and Relationship Management
    {
        "name": "Create Node with Properties",
        "description": "Create a single node with specified properties.",
        "cypher": "CREATE (n:Label {property1: $value1, property2: $value2}) RETURN n",
        "tags": ["create", "node", "basic"],
        "variables": ["value1", "value2"],
    },
    {
        "name": "Merge Node (Upsert)",
        "description": "Merge a node on a unique property, updating timestamps.",
        "cypher": """
MERGE (n:Label {uniqueProperty: $value})
ON CREATE SET n.created = datetime(), n.property2 = $property2
ON MATCH SET n.lastUpdated = datetime(), n.property2 = $property2
RETURN n
        """.strip(),
        "tags": ["merge", "upsert", "node"],
        "variables": ["value", "property2"],
    },
    {
        "name": "Create Relationship",
        "description": "Create a relationship between two existing nodes.",
        "cypher": """
MATCH (a:LabelA {identifier: $value1})
MATCH (b:LabelB {identifier: $value2})
CREATE (a)-[:RELATIONSHIP_TYPE {property: $value}]->(b)
        """.strip(),
        "tags": ["create", "relationship", "basic"],
        "variables": ["value1", "value2", "value"],
    },
    {
        "name": "Update Node Properties",
        "description": "Update specific properties of a matched node.",
        "cypher": """
MATCH (n:Label {identifier: $value})
SET n.property1 = $newValue1, n.property2 = $newValue2
RETURN n
        """.strip(),
        "tags": ["update", "node", "properties"],
        "variables": ["value", "newValue1", "newValue2"],
    },
    {
        "name": "Delete Node and Relationships",
        "description": "Delete a node and all its attached relationships.",
        "cypher": """
MATCH (n:Label {identifier: $value})
DETACH DELETE n
        """.strip(),
        "tags": ["delete", "node", "basic"],
        "variables": ["value"],
    },
    # Querying Patterns
    {
        "name": "Match Node by Property",
        "description": "Find nodes matching a specific property value.",
        "cypher": "MATCH (n:Label) WHERE n.property = $value RETURN n",
        "tags": ["read", "match", "basic"],
        "variables": ["value"],
    },
    {
        "name": "Shortest Path",
        "description": "Find the shortest path between two nodes.",
        "cypher": """
MATCH p = shortestPath((start:Label {identifier: $startValue})-[:RELATIONSHIP_TYPE*]->(end:Label {identifier: $endValue}))
RETURN p
        """.strip(),
        "tags": ["read", "path", "advanced"],
        "variables": ["startValue", "endValue"],
    },
    # Workflow Patterns
    {
        "name": "Log Workflow Execution (FIX)",
        "description": "Log a 'FIX' workflow execution linked to a project and changed files.",
        "cypher": """
MATCH (p:Project {id: $projectId})
CREATE (w:WorkflowExecution {
  id: randomUUID(),
  timestamp: datetime(),
  summary: $summary,
  actionKeyword: 'FIX',
  notes: $notes
})
CREATE (p)-[:HAS_WORKFLOW]->(w)
WITH w
UNWIND $filesChanged AS file
CREATE (f:File {path: file})
CREATE (w)-[:MODIFIED]->(f)
        """.strip(),
        "tags": ["workflow", "logging", "fix"],
        "variables": ["projectId", "summary", "notes", "filesChanged"],
    },
    {
        "name": "Log Workflow Execution (FEATURE)",
        "description": "Log a 'FEATURE' workflow execution linked to a project and changed files.",
        "cypher": """
MATCH (p:Project {id: $projectId})
CREATE (w:WorkflowExecution {
  id: randomUUID(),
  timestamp: datetime(),
  summary: $summary,
  actionKeyword: 'FEATURE',
  notes: $notes
})
CREATE (p)-[:HAS_WORKFLOW]->(w)
WITH w
UNWIND $filesChanged AS file
CREATE (f:File {path: file})
CREATE (w)-[:MODIFIED]->(f)
        """.strip(),
        "tags": ["workflow", "logging", "feature"],
        "variables": ["projectId", "summary", "notes", "filesChanged"],
    },
    # Incarnation Specific - Code Analysis
    {
        "name": "Find Complex Functions",
        "description": "Find functions with high cyclomatic complexity.",
        "cypher": """
MATCH (file:CodeFile)-[:HAS_ANALYSIS]->(analysis:Analysis)
MATCH (analysis)-[:CONTAINS]->(node:ASTNode)
WHERE node.nodeType = 'function_definition' AND node.complexity > 10
RETURN file.path, node.name, node.complexity
ORDER BY node.complexity DESC
        """.strip(),
        "tags": ["analysis", "complexity", "code"],
        "variables": [],
    },
    # Incarnation Specific - Research
    {
        "name": "Create Hypothesis",
        "description": "Create a new scientific hypothesis.",
        "cypher": """
CREATE (h:Hypothesis {
  id: randomUUID(),
  text: $text,
  description: $description,
  priorProbability: $probability,
  status: 'Active',
  created: datetime()
})
RETURN h.id
        """.strip(),
        "tags": ["research", "hypothesis", "create"],
        "variables": ["text", "description", "probability"],
    },
    # Incarnation Specific - Knowledge Graph
    {
        "name": "Find Entities by Content",
        "description": "Find entities based on observation content.",
        "cypher": """
MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
WHERE o.content CONTAINS $searchTerm
RETURN e.name, e.entityType, o.content
        """.strip(),
        "tags": ["knowledge-graph", "search", "entity"],
        "variables": ["searchTerm"],
    },
]


async def seed_snippets() -> None:
    # Get connection details from environment
    uri = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    logger.info(f"Connecting to Neo4j at {uri}...")

    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

        async with driver.session() as session:
            # First, ensure constants/constraints exist (optional but good practice)
            await session.run(
                "CREATE CONSTRAINT snippet_id IF NOT EXISTS FOR (s:CypherSnippet) REQUIRE s.id IS UNIQUE"
            )

            logger.info(f"Seeding {len(SNIPPETS)} snippets...")

            for snippet in SNIPPETS:
                # Generate a deterministic ID based on name to allow updates/idempotency
                snippet_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(snippet["name"])))

                query = """
                MERGE (s:CypherSnippet {id: $id})
                SET s.name = $name,
                    s.description = $description,
                    s.cypher = $cypher,
                    s.variables = $variables,
                    s.lastUpdated = datetime()

                WITH s
                // Clear existing tags
                OPTIONAL MATCH (s)-[r:HAS_TAG]->(t:Tag)
                DELETE r

                WITH s
                UNWIND $tags as tagName
                MERGE (t:Tag {name: tagName})
                MERGE (s)-[:HAS_TAG]->(t)

                RETURN s.name
                """

                await session.run(
                    query,
                    {
                        "id": snippet_id,
                        "name": snippet["name"],
                        "description": snippet["description"],
                        "cypher": snippet["cypher"],
                        "tags": snippet["tags"],
                        "variables": snippet.get("variables", []),
                    },
                )
                logger.info(f"Upserted snippet: {snippet['name']}")

        await driver.close()
        logger.info("Seeding complete!")

    except Exception as e:
        logger.error(f"Error seeding snippets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(seed_snippets())
