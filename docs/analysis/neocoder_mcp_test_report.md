# NeoCoder MCP Tool Verification Report

**Date:** December 16, 2025
**Tester:** AntiGravity Agent
**Topic:** System Integrity & Tool Reliability Analysis

---

## 1. Executive Summary

This report details the comprehensive deep-testing of the NeoCoder Model Context Protocol (MCP) server integration. Following the principles of the **Recursive Domain Construction Framework (RDCF)**, specifically the need for robust "Adversarial Governance" and a reliable "Constraint Ledger," we critically evaluated the tools available to the "Architect" and "Shredder" personas.

**Overall Status:** ⚠️ **DEGRADED INTEGRITY**

While the core read capabilities and high-level workflow orchestrations are functional, critical write operations for the "Constraint Ledger" (Cypher Snippets) and Incarnation switching exhibit significant failures in error handling and response serialization. These issues effectively "blind" the agent to the success of its own actions, increasing the drift risk defined in RDCF Section 2.1.

---

## 2. Critical Findings (The "Shredder" Analysis)

### 2.1 🔴 Serialization Failures in Write Operations (Blind Writes)
The system suffers from a critical serialization bug in write-heavy tools. The underlying database operations succeed, but the MCP server fails to report success, instead throwing an internal Python error.

*   **Affected Tools:** `create_cypher_snippet`, `delete_cypher_snippet`
*   **Error:** `Error: the JSON object must be str, bytes or bytearray, not ResultSummary`
*   **Impact:** The agent believes the operation failed when it actually succeeded. This causes state desynchronization—the agent may retry creates (causing duplicates) or assume data wasn't deleted.
*   **RDCF Implications:** The "Constraint Ledger" becomes unreliable if the agent cannot verify its writes.

### 2.2 🟠 Incarnation Switching Conflict
Switching to the `knowledge_graph` incarnation fails due to improper idempotency in schema initialization.

*   **Tool:** `switch_incarnation(incarnation_type="knowledge_graph")`
*   **Error:** `Neo.ClientError.Schema.IndexAlreadyExists`
*   **Impact:** The server attempts to re-create existing indices/constraints instead of verifying their existence. While the internal state pointer updates, the initialization sequence aborts.
*   **RDCF Implications:** Inability to reliably switch to the "Knowledge Graph" persona hampers the system's ability to maintain the comprehensive graph view required for "Cognitive Mosaicking."

---

## 3. Operational Capabilities (The "Architect" View)

Despite the write-side faults, the read-side and structural tools function robustly.

### 3.1 ✅ Healthy Components
*   **Guidance Hub & Navigation:** `get_guidance_hub` provides accurate, context-aware routing instructions.
*   **Project Management:** `list_projects` and `get_project` correctly identify managed repositories.
*   **Action Templates:** The "Standard Operating Procedures" (FIX, SHRED, FEATURE) are fully retrievable.
*   **Tool Requests:** The feature request workflow (`request_tool`, `list_tool_requests`) works perfectly, including proper ID generation and status tracking.
*   **Custom Queries:** Ad-hoc Cypher execution (`run_custom_query`) is functional, allowing manual verification of state when tools fail.

### 3.2 ⚠️ Functionality Verification Table

| Tool Category | Tool Name | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Core** | `check_connection` | ✅ Pass | Database connection stable. |
| | `get_guidance_hub` | ✅ Pass | Returns correct context. |
| **Incarnations** | `list_incarnations` | ✅ Pass | Lists all personas. |
| | `switch_incarnation` | ❌ Fail | Fails on index conflict (Idempotency bug). |
| **Memory** | `create_cypher_snippet` | ❌ Fail | Serialization error (Action succeeds, Report fails). |
| | `delete_cypher_snippet` | ❌ Fail | Serialization error (Action succeeds, Report fails). |
| | `search_cypher_snippets` | ✅ Pass | Accurate retrieval. |
| **Workflow** | `list_action_templates` | ✅ Pass | Correct versioning and schema. |
| | `request_tool` | ✅ Pass | Correct creation and ID return. |

---

## 4. Recommendations for Engineering

To restore full RDCF compliance, we recommend the following patches:

1.  **Fix Result Serialization:** Update the `CypherSnippetMixin` (and likely `PolymorphicAdapterMixin`) to extract the summary execution metrics (e.g., `counters`) from the Neo4j `ResultSummary` object before attempting to return it as JSON. *Do not return raw driver objects.*
2.  **Enforce Idempotency in Init:** Modify the `knowledge_graph` incarnation's initialization logic to checking for `IF NOT EXISTS` or catching the `IndexAlreadyExists` exception during schema setup.
3.  **Enhance Error Wrapping:** The server needs a "safe mode" wrapper that catches internal serialization errors and attempts to check the database state (e.g., "Write appeared to fail serialization, but verifying object existence...") to prevent partial state hallucinations.

---

**Conclusion:** The "Brain" is functional but has a broken proprioception loop. It can move (write data), but it cannot feel its own movement (get success confirmation).
