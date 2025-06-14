"""
Base Incarnation Module for NeoCoder Framework

This module defines the base class for all incarnations, with common functionality
and required interface methods that must be implemented by each incarnation.

The design follows a plugin architecture where different incarnations can be dynamically
discovered and loaded without requiring central registration of types.
"""

import json
import logging
from typing import List

import mcp.types as types
from neo4j import AsyncDriver, AsyncTransaction

logger = logging.getLogger("mcp_neocoder.incarnations.base")


class BaseIncarnation:
    """Base class for all incarnation implementations."""

    # This should be overridden by each incarnation
    name = "base"  # String identifier, should be unique
    description = "Base incarnation class"

    # Optional schema creation scripts, format: List of Cypher queries to execute
    schema_queries: List[str] = []

    # Hub content - default guidance hub text for this incarnation
    hub_content: str = """
# Default Incarnation Hub

Welcome to this incarnation of the NeoCoder framework.
This is a default hub that should be overridden by each incarnation.
    """

    # Optional list of tool method names - can be defined by subclasses
    # to explicitly declare which methods should be registered as tools
    # Format: list[str] with method names to be registered as tools
    _tool_methods: List[str] = []

    def __init__(self, driver: AsyncDriver, database: str = "neo4j"):
        """Initialize the incarnation with database connection."""
        self.driver = driver
        self.database = database

    async def initialize_schema(self):
        """Initialize the Neo4j schema for this incarnation."""
        from ..event_loop_manager import safe_neo4j_session
        
        # Execute schema queries if defined
        if self.schema_queries:
            try:
                async with safe_neo4j_session(self.driver, self.database) as session:
                    # Execute each constraint/index query individually
                    for query in self.schema_queries:
                        await session.execute_write(lambda tx: tx.run(query))  # type: ignore

                # Create guidance hub if needed
                await self.ensure_hub_exists()

                logger.info(f"{self.name} incarnation schema initialized")
            except Exception as e:
                logger.error(f"Error initializing schema for {self.name}: {e}")
                raise
        else:
            # No schema queries defined
            logger.warning(f"No schema queries defined for {self.name}")

            # Still create the hub
            await self.ensure_hub_exists()

    async def ensure_hub_exists(self):
        """Create the guidance hub for this incarnation if it doesn't exist."""
        from ..event_loop_manager import safe_neo4j_session
        
        hub_id = f"{self.name}_hub"

        query = """
        MERGE (hub:AiGuidanceHub {id: $hub_id})
        ON CREATE SET hub.description = $description
        RETURN hub
        """

        params = {
            "hub_id": hub_id,
            "description": self.hub_content
        }

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                await session.execute_write(lambda tx: tx.run(query, params))
                logger.info(f"Ensured hub exists for {self.name}")
        except Exception as e:
            logger.error(f"Error creating hub for {self.name}: {e}")
            raise

    async def get_guidance_hub(self) -> List[types.TextContent]:
        """Get the guidance hub content for this incarnation."""
        from ..event_loop_manager import safe_neo4j_session
        
        hub_id = f"{self.name}_hub"

        query = """
        MATCH (hub:AiGuidanceHub {{id: $hub_id}})
        RETURN hub.description AS description
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_read(lambda tx: self._read_query(tx, query, {"hub_id": hub_id}))  # type: ignore
                results = json.loads(results_json)

                if results and len(results) > 0:
                    return [types.TextContent(type="text", text=results[0]["description"])]
                else:
                    # If hub doesn't exist, create it
                    await self.ensure_hub_exists()
                    # Try again
                    return await self.get_guidance_hub()
        except Exception as e:
            logger.error(f"Error retrieving guidance hub for {self.name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    def list_tool_methods(self):
        """List all methods in this class that appear to be tools.

        Returns:
            list: List of method names that appear to be tools
        """
        import inspect
        import mcp.types as types

        tool_methods = []

        logger.debug(f"Checking methods in {self.__class__.__name__}")

        # First, check if there's a hardcoded base list of tools
        if hasattr(self, '_tool_methods') and isinstance(self._tool_methods, list):
            logger.info(f"Using predefined tool list for {self.__class__.__name__}: {self._tool_methods}")
            for name in self._tool_methods:
                if hasattr(self, name) and callable(getattr(self, name)):
                    tool_methods.append(name)
                else:
                    logger.warning(f"Predefined tool method {name} does not exist in {self.__class__.__name__}")
            return tool_methods

        # If no predefined list, use inspection to find tools
        class_dict = {}

        # Collect methods from all parent classes
        for cls in self.__class__.__mro__:
            if cls is object:
                continue
            class_dict.update(cls.__dict__)

        # Skip these common non-tool methods
        excluded_methods = {
            'initialize_schema', 'get_guidance_hub', 'register_tools',
            'list_tool_methods', 'ensure_hub_exists', '_read_query', '_write'
        }

        for name, method_obj in class_dict.items():
            # Skip private methods and excluded methods
            if name.startswith('_') or name in excluded_methods:
                continue

            # Check if it's an async method
            if inspect.iscoroutinefunction(method_obj):
                # Get the actual bound method
                method = getattr(self, name)

                # Check return type annotation if available
                is_tool = False

                if hasattr(method, '__annotations__'):
                    return_type = method.__annotations__.get('return')
                    # Check for List[types.TextContent] return type
                    if return_type and (
                        return_type == List[types.TextContent] or
                        getattr(return_type, '__origin__', None) is list and
                        getattr(return_type, '__args__', [None])[0] == types.TextContent
                    ):
                        is_tool = True
                        logger.debug(f"Identified tool method via return type annotation: {name}")

                # Fallback: if it's an async method defined in the class itself (not inherited),
                # and it has parameters, assume it's a tool
                if not is_tool and hasattr(method, '__code__'):
                    # Check if it has at least one parameter beyond 'self'
                    if method.__code__.co_argcount > 1:
                        is_tool = True
                        logger.debug(f"Identified tool method via parameter count: {name}")

                if is_tool:
                    tool_methods.append(name)

        logger.info(f"Found {len(tool_methods)} tool methods in {self.__class__.__name__} via inspection: {tool_methods}")
        return tool_methods

    # Track registered tools at the class level - using a class variable
    _registered_tool_methods = set()

    async def register_tools(self, server):
        """Identify tool methods and register them with the central ToolRegistry."""
        # Get all tool methods from this incarnation
        tool_methods = self.list_tool_methods()
        logger.info(f"Identified tool methods in {self.name}: {tool_methods}")

        # Register these tools with the central tool registry for tracking/listing
        from ..tool_registry import registry as tool_registry

        # Let register_class_tools handle adding to the registry's internal structures
        tools_added_to_registry_count = tool_registry.register_class_tools(self, self.name)

        # Log based on tools found and added to the registry
        logger.info(f"{self.name} incarnation: {tools_added_to_registry_count} tools added to ToolRegistry")

        # Return the count of tools identified/added to registry
        return len(tool_methods)

    import typing

    async def _read_query(self, tx: AsyncTransaction, query: typing.LiteralString, params: dict) -> str:
        """Execute a read query and return results as JSON string."""
        result = await tx.run(query, params)
        records = await result.data()  # Use .data() instead of .to_eager_result().records
        return json.dumps(records, default=str)

    async def _write(self, tx: AsyncTransaction, query: typing.LiteralString, params: dict):
        """Execute a write query and return results as JSON string."""
        result = await tx.run(query, params or {})
        summary = await result.consume()
        return summary

    async def safe_session(self):
        """Get a safe Neo4j session that handles event loop issues.
        
        This is a convenience method that incarnations can use to get a session
        that properly handles asyncio event loop management.
        
        Usage:
            async with self.safe_session() as session:
                # Use session here
        """
        from ..event_loop_manager import safe_neo4j_session
        return safe_neo4j_session(self.driver, self.database)


# End of base incarnation module
