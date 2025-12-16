from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")


def my_tool() -> None:
    pass


mcp.add_tool(my_tool)

print("Attributes of FastMCP instance:")
for attr in dir(mcp):
    if not attr.startswith("__"):
        print(f"- {attr}")

print("\nScanning for underlying tool storage...")
# Check common locations for tool storage in FastMCP implementations
if hasattr(mcp, "_tools"):
    print(f"Found _tools: {mcp._tools}")
if hasattr(mcp, "tools"):
    print(f"Found tools: {mcp.tools}")
if hasattr(mcp, "_tool_manager"):
    print(f"Found _tool_manager: {mcp._tool_manager}")

# Check internal structure of tool manager
print("\nAttributes of ToolManager:")
tm = mcp._tool_manager
for attr in dir(tm):
    if not attr.startswith("__"):
        print(f"- {attr}")

print("\nScanning ToolManager storage...")
if hasattr(tm, "_tools"):
    print(
        f"Found _tools in manager: keys={list(tm._tools.keys()) if isinstance(tm._tools, dict) else tm._tools}"
    )
if hasattr(tm, "tools"):
    print(f"Found tools in manager: {tm.tools}")
