"""
Event Loop Manager for NeoCoder

This module provides tools to manage Neo4j sessions consistently.
Simplified to rely on standard asyncio and Neo4j driver capabilities.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from neo4j import AsyncDriver

logger = logging.getLogger("mcp_neocoder")


@asynccontextmanager
async def safe_neo4j_session(
    driver: AsyncDriver, database: str
) -> AsyncGenerator[Any, None]:
    """
    Create a Neo4j session ensuring proper tracking.

    This context manager wraps the standard driver.session() to ensure
    sessions are tracked by the process manager for cleanup.
    """
    # Import tracking functions
    from .process_manager import track_session, untrack_session

    session = None
    try:
        # Create session directly using the driver
        # The Neo4j 5.x+ driver handles async context natively
        session = driver.session(database=database)

        # Track the session for cleanup
        track_session(session)

        async with session as s:
            yield s

    except Exception as e:
        logger.error(f"Error in Neo4j session: {e}")
        # Add context if it looks like a loop issue, though these should be rarer now
        if "attached to a different loop" in str(e):
            logger.error(
                "Event loop mismatch detected. Ensure driver and session are used in the same async context."
            )
        raise
    finally:
        # Always untrack the session
        if session:
            untrack_session(session)


def initialize_main_loop() -> asyncio.AbstractEventLoop:
    """Initialize and return the main event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
