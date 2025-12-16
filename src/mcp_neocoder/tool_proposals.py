"""
Tool Proposal System for NeoCoder Neo4j AI Workflow

This module provides functionality for AI assistants to propose new tools
and for users to request new tool capabilities in the NeoCoder system.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import mcp.types as types
from neo4j import AsyncManagedTransaction
from pydantic import Field

from .event_loop_manager import safe_neo4j_session

logger = logging.getLogger("mcp_neocoder.tool_proposals")


# Constants for Field definitions to avoid B008
_TOOL_NAME_FIELD = Field(..., description="Proposed tool name")
_PROPOSAL_DESCRIPTION_FIELD = Field(
    ..., description="Description of the tool's functionality"
)
_PARAMETERS_FIELD = Field(..., description="List of parameter definitions for the tool")
_RATIONALE_FIELD = Field(
    ..., description="Rationale for why this tool would be valuable"
)
_IMPLEMENTATION_NOTES_FIELD = Field(
    None, description="Optional technical notes for implementation"
)
_EXAMPLE_USAGE_FIELD = Field(
    None, description="Optional example of how the tool would be used"
)

_REQUEST_DESCRIPTION_FIELD = Field(
    ..., description="Description of the desired tool functionality"
)
_USE_CASE_FIELD = Field(..., description="How you would use this tool")
_PRIORITY_FIELD = Field(
    "MEDIUM", description="Priority of the request (LOW, MEDIUM, HIGH)"
)
_REQUESTED_BY_FIELD = Field(None, description="Name of the person requesting the tool")

_PROPOSAL_ID_FIELD = Field(..., description="ID of the tool proposal to retrieve")
_REQUEST_ID_FIELD = Field(..., description="ID of the tool request to retrieve")

_PROPOSAL_STATUS_FILTER_FIELD = Field(
    None, description="Filter by status (Proposed, Approved, Implemented, Rejected)"
)
_PROPOSAL_LIMIT_FIELD = Field(10, description="Maximum number of proposals to return")

_REQUEST_STATUS_FILTER_FIELD = Field(
    None, description="Filter by status (Submitted, In Review, Implemented, Rejected)"
)
_REQUEST_PRIORITY_FILTER_FIELD = Field(
    None, description="Filter by priority (LOW, MEDIUM, HIGH)"
)
_REQUEST_LIMIT_FIELD = Field(10, description="Maximum number of requests to return")


class ToolProposalMixin:
    """Mixin class providing tool proposal functionality for the Neo4jWorkflowServer."""

    database: str = "neo4j"
    driver: Any = None

    async def _read_query(
        self, tx: AsyncManagedTransaction, query: str, params: dict
    ) -> str:
        """Execute a read query and return results as JSON string."""
        raise NotImplementedError("_read_query must be implemented by the parent class")

    async def _write(
        self, tx: AsyncManagedTransaction, query: str, params: dict
    ) -> str:
        """Execute a write query and return results as JSON string."""
        raise NotImplementedError("_write must be implemented by the parent class")

    async def propose_tool(
        self,
        name: str = _TOOL_NAME_FIELD,
        description: str = _PROPOSAL_DESCRIPTION_FIELD,
        parameters: List[Dict[str, Any]] = _PARAMETERS_FIELD,
        rationale: str = _RATIONALE_FIELD,
        implementation_notes: Optional[str] = _IMPLEMENTATION_NOTES_FIELD,
        example_usage: Optional[str] = _EXAMPLE_USAGE_FIELD,
    ) -> List[types.TextContent]:
        """Propose a new tool for the NeoCoder system."""

        # Generate a proposal ID
        proposal_id = str(uuid.uuid4())

        # Organize parameters as a JSON string for storage
        parameters_json = json.dumps(parameters)

        # Build query for creating the proposal
        query = """
        CREATE (p:ToolProposal {
            id: $id,
            name: $name,
            description: $description,
            parameters: $parameters,
            rationale: $rationale,
            timestamp: datetime(),
            status: "Proposed"
        })
        """

        params = {
            "id": proposal_id,
            "name": name,
            "description": description,
            "parameters": parameters_json,
            "rationale": rationale,
        }

        # Add optional fields if provided
        if implementation_notes:
            query = query.replace(
                'status: "Proposed"',
                'status: "Proposed", implementationNotes: $implementationNotes',
            )
            params["implementationNotes"] = implementation_notes

        if example_usage:
            query = query.replace(
                'status: "Proposed"', 'status: "Proposed", exampleUsage: $exampleUsage'
            )
            params["exampleUsage"] = example_usage

        # Complete the query
        query += """
        WITH p
        MATCH (hub:AiGuidanceHub {id: 'main_hub'})
        CREATE (hub)-[:HAS_PROPOSAL]->(p)
        RETURN p.id AS id, p.name AS name
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_write(
                    self._read_query, query, params
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    text = "# Tool Proposal Submitted\n\n"
                    text += "Thank you for proposing a new tool. Your proposal has been recorded.\n\n"
                    text += f"**Proposal ID:** {proposal_id}\n"
                    text += f"**Tool Name:** {name}\n"
                    text += "**Status:** Proposed\n\n"
                    text += f'The proposal will be reviewed by the development team. You can check the status of your proposal using `get_tool_proposal(id="{proposal_id}")`.'

                    return [types.TextContent(type="text", text=text)]
                else:
                    return [
                        types.TextContent(
                            type="text", text="Error submitting tool proposal"
                        )
                    ]
        except Exception as e:
            logger.error(f"Error proposing tool: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def request_tool(
        self,
        description: str = _REQUEST_DESCRIPTION_FIELD,
        use_case: str = _USE_CASE_FIELD,
        priority: str = _PRIORITY_FIELD,
        requested_by: Optional[str] = _REQUESTED_BY_FIELD,
    ) -> List[types.TextContent]:
        """Request a new tool feature for the NeoCoder system."""

        # Generate a request ID
        request_id = str(uuid.uuid4())

        # Build query for creating the tool request
        query = """
        CREATE (r:ToolRequest {
            id: $id,
            description: $description,
            useCase: $useCase,
            priority: $priority,
            timestamp: datetime(),
            status: "Submitted"
        })
        """

        params = {
            "id": request_id,
            "description": description,
            "useCase": use_case,
            "priority": priority,
        }

        # Add requester name if provided
        if requested_by:
            query = query.replace(
                'status: "Submitted"', 'status: "Submitted", requestedBy: $requestedBy'
            )
            params["requestedBy"] = requested_by

        # Complete the query
        query += """
        WITH r
        MATCH (hub:AiGuidanceHub {id: 'main_hub'})
        CREATE (hub)-[:HAS_REQUEST]->(r)
        RETURN r.id AS id, r.description AS description
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_write(
                    self._read_query, query, params
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    text = "# Tool Request Submitted\n\n"
                    text += "Thank you for requesting a new tool. Your request has been recorded.\n\n"
                    text += f"**Request ID:** {request_id}\n"
                    text += f"**Description:** {description}\n"
                    text += f"**Priority:** {priority}\n"
                    text += "**Status:** Submitted\n\n"
                    text += f'The request will be reviewed by the development team. You can check the status of your request using `get_tool_request(id="{request_id}")`.'

                    return [types.TextContent(type="text", text=text)]
                else:
                    return [
                        types.TextContent(
                            type="text", text="Error submitting tool request"
                        )
                    ]
        except Exception as e:
            logger.error(f"Error requesting tool: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def get_tool_proposal(
        self, id: str = _PROPOSAL_ID_FIELD
    ) -> List[types.TextContent]:
        """Get a specific tool proposal by ID."""
        query = """
        MATCH (p:ToolProposal {id: $id})
        RETURN p.id AS id,
               p.name AS name,
               p.description AS description,
               p.parameters AS parameters,
               p.rationale AS rationale,
               p.timestamp AS timestamp,
               p.status AS status,
               p.implementationNotes AS implementationNotes,
               p.exampleUsage AS exampleUsage
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_read(
                    self._read_query, query, {"id": id}
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    proposal = results[0]

                    # Parse parameters from JSON
                    parameters = []
                    if proposal.get("parameters"):
                        try:
                            parameters = json.loads(proposal["parameters"])
                        except json.JSONDecodeError:
                            parameters = [{"error": "Could not parse parameters"}]

                    text = f"# Tool Proposal: {proposal.get('name', 'Unnamed')}\n\n"
                    text += f"**ID:** {proposal.get('id', id)}\n"
                    text += f"**Status:** {proposal.get('status', 'Unknown')}\n"
                    text += f"**Submitted:** {proposal.get('timestamp', 'Unknown')}\n\n"

                    text += f"## Description\n\n{proposal.get('description', 'No description')}\n\n"
                    text += f"## Rationale\n\n{proposal.get('rationale', 'No rationale provided')}\n\n"

                    text += "## Parameters\n\n"
                    if parameters:
                        for i, param in enumerate(parameters, 1):
                            text += (
                                f"### {i}. {param.get('name', 'Unnamed parameter')}\n"
                            )
                            text += (
                                f"- **Type:** {param.get('type', 'Not specified')}\n"
                            )
                            text += f"- **Description:** {param.get('description', 'No description')}\n"
                            text += (
                                f"- **Required:** {param.get('required', False)}\n\n"
                            )
                    else:
                        text += "No parameters defined.\n\n"

                    if proposal.get("exampleUsage"):
                        text += f"## Example Usage\n\n{proposal['exampleUsage']}\n\n"

                    if proposal.get("implementationNotes"):
                        text += f"## Implementation Notes\n\n{proposal['implementationNotes']}\n\n"

                    return [types.TextContent(type="text", text=text)]
                else:
                    return [
                        types.TextContent(
                            type="text", text=f"No tool proposal found with ID '{id}'"
                        )
                    ]
        except Exception as e:
            logger.error(f"Error retrieving tool proposal: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def get_tool_request(
        self, id: str = _REQUEST_ID_FIELD
    ) -> List[types.TextContent]:
        """Get a specific tool request by ID."""
        query = """
        MATCH (r:ToolRequest {id: $id})
        OPTIONAL MATCH (r)-[:IMPLEMENTED_AS]->(p:ToolProposal)
        RETURN r.id AS id,
               r.description AS description,
               r.useCase AS useCase,
               r.priority AS priority,
               r.timestamp AS timestamp,
               r.status AS status,
               r.requestedBy AS requestedBy,
               p.id AS proposalId,
               p.name AS proposalName
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_read(
                    self._read_query, query, {"id": id}
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    request = results[0]

                    text = "# Tool Request\n\n"
                    text += f"**ID:** {request.get('id', id)}\n"
                    text += f"**Status:** {request.get('status', 'Unknown')}\n"
                    text += f"**Priority:** {request.get('priority', 'MEDIUM')}\n"
                    text += f"**Submitted:** {request.get('timestamp', 'Unknown')}\n"

                    if request.get("requestedBy"):
                        text += f"**Requested By:** {request['requestedBy']}\n"

                    text += f"\n## Description\n\n{request.get('description', 'No description')}\n\n"
                    text += f"## Use Case\n\n{request.get('useCase', 'No use case provided')}\n\n"

                    if request.get("proposalId"):
                        text += "## Implementation\n\n"
                        text += f"This request has been implemented as the tool proposal '{request.get('proposalName', 'Unnamed')}'.\n"
                        text += f"You can view the full proposal with `get_tool_proposal(id=\"{request['proposalId']}\")`\n"

                    return [types.TextContent(type="text", text=text)]
                else:
                    return [
                        types.TextContent(
                            type="text", text=f"No tool request found with ID '{id}'"
                        )
                    ]
        except Exception as e:
            logger.error(f"Error retrieving tool request: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def list_tool_proposals(
        self,
        status: Optional[str] = _PROPOSAL_STATUS_FILTER_FIELD,
        limit: int = _PROPOSAL_LIMIT_FIELD,
    ) -> List[types.TextContent]:
        """List all tool proposals with optional filtering."""
        query = """
        MATCH (p:ToolProposal)
        WHERE 1=1
        """

        params: Dict[str, Any] = {"limit": limit}

        if status:
            query += " AND p.status = $status"
            params["status"] = status

        query += """
        RETURN p.id AS id,
               p.name AS name,
               p.description AS description,
               p.timestamp AS timestamp,
               p.status AS status
        ORDER BY p.timestamp DESC
        LIMIT $limit
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_read(
                    self._read_query, query, params
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    status_filter = f" ({status})" if status else ""

                    text = f"# Tool Proposals{status_filter}\n\n"
                    text += "| ID | Name | Status | Submitted | Description |\n"
                    text += "| -- | ---- | ------ | --------- | ----------- |\n"

                    for p in results:
                        text += f"| {p.get('id', 'N/A')[:8]}... | {p.get('name', 'Unnamed')} | {p.get('status', 'Unknown')} | {p.get('timestamp', 'Unknown')[:10]} | {p.get('description', 'No description')[:50]}... |\n"

                    text += '\nTo view full details of a proposal, use `get_tool_proposal(id="proposal-id")`'

                    return [types.TextContent(type="text", text=text)]
                else:
                    status_msg = f" with status '{status}'" if status else ""
                    return [
                        types.TextContent(
                            type="text", text=f"No tool proposals found{status_msg}."
                        )
                    ]
        except Exception as e:
            logger.error(f"Error listing tool proposals: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def list_tool_requests(
        self,
        status: Optional[str] = _REQUEST_STATUS_FILTER_FIELD,
        priority: Optional[str] = _REQUEST_PRIORITY_FILTER_FIELD,
        limit: int = _REQUEST_LIMIT_FIELD,
    ) -> List[types.TextContent]:
        """List all tool requests with optional filtering."""
        query = """
        MATCH (r:ToolRequest)
        WHERE 1=1
        """

        params: Dict[str, Any] = {"limit": limit}

        if status:
            query += " AND r.status = $status"
            params["status"] = status

        if priority:
            query += " AND r.priority = $priority"
            params["priority"] = priority

        query += """
        RETURN r.id AS id,
               r.description AS description,
               r.priority AS priority,
               r.timestamp AS timestamp,
               r.status AS status,
               r.requestedBy AS requestedBy
        ORDER BY
            CASE r.priority
                WHEN 'HIGH' THEN 1
                WHEN 'MEDIUM' THEN 2
                WHEN 'LOW' THEN 3
                ELSE 4
            END,
            r.timestamp DESC
        LIMIT $limit
        """

        try:
            async with safe_neo4j_session(self.driver, self.database) as session:
                results_json = await session.execute_read(
                    self._read_query, query, params
                )
                results = json.loads(results_json)

                if results and len(results) > 0:
                    filters = []
                    if status:
                        filters.append(f"Status: {status}")
                    if priority:
                        filters.append(f"Priority: {priority}")

                    filter_text = f" ({', '.join(filters)})" if filters else ""

                    text = f"# Tool Requests{filter_text}\n\n"
                    text += "| ID | Priority | Status | Submitted | Description |\n"
                    text += "| -- | -------- | ------ | --------- | ----------- |\n"

                    for r in results:
                        text += f"| {r.get('id', 'N/A')[:8]}... | {r.get('priority', 'MEDIUM')} | {r.get('status', 'Unknown')} | {r.get('timestamp', 'Unknown')[:10]} | {r.get('description', 'No description')[:50]}... |\n"

                    text += '\nTo view full details of a request, use `get_tool_request(id="request-id")`'

                    return [types.TextContent(type="text", text=text)]
                else:
                    filters = []
                    if status:
                        filters.append(f"status '{status}'")
                    if priority:
                        filters.append(f"priority '{priority}'")

                    filter_text = f" with {' and '.join(filters)}" if filters else ""
                    return [
                        types.TextContent(
                            type="text", text=f"No tool requests found{filter_text}."
                        )
                    ]
        except Exception as e:
            logger.error(f"Error listing tool requests: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]
