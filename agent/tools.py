from langchain_core.tools import tool
from typing import Annotated, List
from logs.log import logger
from metrics.prometheus import track_tool_execution, track_error
from langgraph.prebuilt import InjectedState
from agent.context_offloader import context_offloader
from pydantic import BaseModel, Field
import time
import hashlib
import json


def hash_query(query: str) -> str:
    """Hash query for logging (privacy)"""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


class GetChunksInput(BaseModel):
    """Input schema for chunk retrieval"""
    session_id: str = Field(description="Session ID from database query")
    chunk_indices: List[int] = Field(
        description="List of chunk indices to retrieve (e.g., [0] for first, [1,2] for next)"
    )


@tool
async def query_database_with_offload(
    query: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Query database and offload large results to in-memory chunks.
    
    For large result sets (>100 rows), data is chunked and stored.
    Use get_context_chunks tool to retrieve data in manageable pieces.
    
    Args:
        query: SQL query to execute
        state: Injected state with access tokens
        
    Returns:
        JSON with session_id for chunk retrieval or direct data if small
    """
    start_time = time.time()
    query_hash = hash_query(query)
    
    try:
        from database.client import run_query
        
        # FIXED: Safely extract config from state
        config = state.get("config", {})
        if not config:
            # Fallback: check if tokens are at root level
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
        else:
            access_token = config.get("access_token")
            refresh_token = config.get("refresh_token")
        
        if not access_token:
            logger.error("tool_error - tool=query_database, error=no_access_token")
            track_tool_execution("query_database_with_offload", 0, success=False)
            track_error("missing_access_token", "tool")
            return json.dumps({"error": "No access token. Please authenticate."})
        
        logger.info(f"tool_exec_start - tool=query_database, query_hash={query_hash}")
        
        # Execute query
        results = await run_query(
            query=query,
            access_token=access_token,
            refresh_token=refresh_token
        )
        
        duration = time.time() - start_time
        track_tool_execution("query_database_with_offload", duration, success=True)
        
        # No results
        if not results:
            return json.dumps({
                "status": "no_data",
                "message": "No records found matching your query."
            })
        
        # Small result set - return directly
        if len(results) <= 2:
            logger.info(
                f"tool_exec_complete - tool=query_database, query_hash={query_hash}, "
                f"rows={len(results)}, duration_ms={duration*1000:.2f}, offload=no"
            )
            return json.dumps({
                "status": "success",
                "rows": len(results),
                "data": results,
                "message": f"Found {len(results)} records."
            })
        
        # Large result set - offload to memory
        session_id = context_offloader.store(
            rows=results,
            query=query,
            year_filter=None  # Can extract from query if needed
        )
        
        metadata = context_offloader.get_metadata(session_id)
        summary = context_offloader.get_summary(session_id)
        
        # FIX: Remove debug print, use proper logging
        logger.info(f"context_offload_summary - session_id={session_id}, summary={json.dumps(summary)}")
        
        logger.info(
            f"tool_exec_complete - tool=query_database, query_hash={query_hash}, "
            f"rows={len(results)}, chunks={metadata['total_chunks']}, "
            f"duration_ms={duration*1000:.2f}, offload=yes"
        )
        
        return json.dumps({
            "status": "offloaded",
            "message": (
                f"Found {len(results)} records. Data has been chunked into "
                f"{metadata['total_chunks']} chunks for efficient retrieval."
            ),
            "session_id": session_id,
            "total_rows": len(results),
            "total_chunks": metadata["total_chunks"],
            "summary": summary,
            "next_step": (
                f"Use get_context_chunks tool with session_id='{session_id}' "
                f"and chunk_indices=[0] to start retrieving data."
            )
        }, indent=2)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"tool_exec_error - tool=query_database, query_hash={query_hash}, "
            f"error={e}, duration_ms={duration*1000:.2f}",
            exc_info=True
        )
        track_tool_execution("query_database_with_offload", duration, success=False)
        track_error("tool_execution_error", "tool")
        return json.dumps({"error": f"Query execution failed: {str(e)}"})


@tool(args_schema=GetChunksInput)
async def get_context_chunks(
    session_id: str,
    chunk_indices: List[int],
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Retrieve specific chunks from offloaded context.
    
    IMPORTANT: Returns ONLY the requested chunks.
    - Start with chunk_indices=[0] for first chunk
    - Request more as needed based on remaining_chunk_indices
    - Consider token budget before requesting multiple chunks
    
    Args:
        session_id: Session ID from database query tool
        chunk_indices: Specific chunks to retrieve (e.g., [0], [1,2])
        state: Injected state
        
    Returns:
        JSON with requested chunk data
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"tool_exec_start - tool=get_chunks, session_id={session_id}, "
            f"chunks={chunk_indices}"
        )
        
        result = context_offloader.get_chunks(session_id, chunk_indices)
        
        duration = time.time() - start_time
        
        if result["status"] == "error":
            track_tool_execution("get_context_chunks", duration, success=False)
            logger.warning(
                f"tool_exec_error - tool=get_chunks, session_id={session_id}, "
                f"error={result['message']}"
            )
        else:
            track_tool_execution("get_context_chunks", duration, success=True)
            logger.info(
                f"tool_exec_complete - tool=get_chunks, session_id={session_id}, "
                f"chunks={chunk_indices}, rows={result['rows_in_response']}, "
                f"duration_ms={duration*1000:.2f}"
            )
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"tool_exec_error - tool=get_chunks, session_id={session_id}, "
            f"error={e}, duration_ms={duration*1000:.2f}",
            exc_info=True
        )
        track_tool_execution("get_context_chunks", duration, success=False)
        track_error("chunk_retrieval_error", "tool")
        return json.dumps({"error": f"Chunk retrieval failed: {str(e)}"})


# List of all tools
TOOLS = [query_database_with_offload, get_context_chunks]