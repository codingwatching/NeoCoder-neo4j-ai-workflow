"""
Process and Resource Management for NeoCoder MCP Server

A lightweight resource tracker to ensure Neo4j drivers and background tasks
are properly cleaned up on server shutdown.
"""

import asyncio
import atexit
import logging
import signal
import sys
from typing import Any, Dict, Set

logger = logging.getLogger("mcp_neocoder")

# Global tracking
background_tasks: Set[asyncio.Task] = set()
active_drivers: Set[Any] = set()
active_sessions: Set[Any] = set()
_cleanup_registered = False


def track_background_task(task: asyncio.Task) -> None:
    """Track a background task for cleanup."""
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


def track_driver(driver: Any) -> None:
    """Track a Neo4j driver for cleanup."""
    active_drivers.add(driver)


def untrack_driver(driver: Any) -> None:
    """Remove a driver from tracking."""
    active_drivers.discard(driver)


def track_session(session: Any) -> None:
    """Track a Neo4j session for cleanup."""
    active_sessions.add(session)


def untrack_session(session: Any) -> None:
    """Remove a session from tracking."""
    active_sessions.discard(session)


async def cleanup_resources() -> None:
    """Clean up all tracked resources."""
    logger.info("Starting resource cleanup...")

    # Cancel background tasks
    if background_tasks:
        logger.info(f"Cancelling {len(background_tasks)} background tasks")
        for task in background_tasks:
            if not task.done():
                task.cancel()

        # Give them a moment to cancel
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    # Close sessions
    if active_sessions:
        logger.info(f"Closing {len(active_sessions)} active sessions")
        for session in list(active_sessions):
            try:
                if hasattr(session, "close"):
                    if asyncio.iscoroutinefunction(session.close):
                        await session.close()
                    else:
                        session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
        active_sessions.clear()

    # Close drivers
    if active_drivers:
        logger.info(f"Closing {len(active_drivers)} active drivers")
        for driver in list(active_drivers):
            try:
                await driver.close()
            except Exception as e:
                logger.error(f"Error closing driver: {e}")
        active_drivers.clear()

    logger.info("Resource cleanup completed")


def cleanup_processes_sync() -> None:
    """Synchronous wrapper for cleanup suitable for atexit/signal handlers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # logic for running loop
        asyncio.create_task(cleanup_resources())
    else:
        try:
            asyncio.run(cleanup_resources())
        except Exception as e:
            logger.error(f"Error during sync cleanup: {e}")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_processes_sync()
    sys.exit(0)


def register_cleanup_handlers() -> None:
    """Register signal handlers and atexit hook."""
    global _cleanup_registered
    if _cleanup_registered:
        return

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(cleanup_processes_sync)
        _cleanup_registered = True
        logger.info("Cleanup handlers registered")
    except Exception as e:
        logger.error(f"Failed to register cleanup handlers: {e}")


def get_cleanup_status() -> Dict[str, Any]:
    """Get status for monitoring."""
    return {
        "running_processes": 0,  # Legacy compatibility
        "background_tasks": len(background_tasks),
        "active_drivers": len(active_drivers),
        "active_sessions": len(active_sessions),
        "cleanup_registered": _cleanup_registered,
        "process_ids": [],  # Legacy compatibility
    }


def cleanup_zombie_instances() -> int:
    """Clean up zombie instances (stub for compatibility)."""
    cleanup_processes_sync()
    return 0
