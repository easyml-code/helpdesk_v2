from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid
from logs.log import logger
from metrics.prometheus import (
    cache_operations_total, 
    track_context_offload,
    track_chunk_retrieval
)


class InMemoryContextOffloader:
    """
    In-memory context offloading for large database results.
    Chunks data and stores in memory (NO REDIS).
    Tracks which chunks have been retrieved by the agent.
    """
    
    def __init__(self, chunk_context_size: int = 2000):
        # Storage: {session_id: {chunks, metadata, summary, retrieval_tracking}}
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.chunk_context_size = chunk_context_size
        logger.info(f"InMemoryContextOffloader initialized - chunk_size={chunk_context_size}")
    
    def _chunk_data(self, rows: List[Dict]) -> List[List[Dict]]:
        """Chunk rows ensuring complete rows within context size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for row in rows:
            row_str = json.dumps(row)
            row_size = len(row_str)
            
            # Single row exceeds context size - put in own chunk
            if row_size > self.chunk_context_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                chunks.append([row])
                continue
            
            # Check if adding row exceeds context size
            if current_size + row_size > self.chunk_context_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [row]
                current_size = row_size
            else:
                current_chunk.append(row)
                current_size += row_size
        
        # Add remaining rows
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _calculate_summary(self, rows: List[Dict], year_filter: Optional[int]) -> Dict[str, Any]:
        """Calculate summary statistics from all rows before chunking"""
        if not rows:
            return {
                "total_invoices": 0,
                "total_amount": 0.0,
                "date_range": {"start": None, "end": None},
                "status_distribution": {},
                "year_filter_applied": year_filter is not None,
                "filtered_year": year_filter
            }
        
        # Calculate totals
        total_amount = sum(row.get("amount", 0) for row in rows)
        
        # Extract dates
        dates = [row.get("date") for row in rows if row.get("date")]
        
        # Status distribution
        status_dist = {}
        for row in rows:
            status = row.get("status", "unknown")
            status_dist[status] = status_dist.get(status, 0) + 1
        
        return {
            "total_invoices": len(rows),
            "total_amount": round(total_amount, 2),
            "date_range": {
                "start": min(dates) if dates else None,
                "end": max(dates) if dates else None
            },
            "status_distribution": status_dist,
            "year_filter_applied": year_filter is not None,
            "filtered_year": year_filter
        }
    
    def store(self, rows: List[Dict], query: str, year_filter: Optional[int] = None) -> str:
        """
        Store rows with chunking in memory.
        Initializes retrieval tracking for agent monitoring.
        
        Returns:
            session_id for retrieval
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        # Chunk the data
        chunks = self._chunk_data(rows)
        
        # Calculate summary BEFORE chunking
        summary = self._calculate_summary(rows, year_filter)
        
        # Initialize retrieval tracking
        chunk_tracking = {
            idx: {
                "retrieved": False,
                "retrieval_count": 0,
                "first_retrieval": None,
                "last_retrieval": None,
                "row_count": len(chunk)
            }
            for idx, chunk in enumerate(chunks)
        }
        
        # Store in memory
        self.storage[session_id] = {
            "session_id": session_id,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "total_rows": len(rows),
            "query": query,
            "year_filter": year_filter,
            "summary": summary,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "chunk_tracking": chunk_tracking,
            "retrieval_history": []  # Log of all retrieval operations
        }
        
        cache_operations_total.labels(operation="set", result="success").inc()
        track_context_offload("store", len(rows), len(chunks), success=True)
        
        logger.info(
            f"context_offload_stored - session_id={session_id}, "
            f"rows={len(rows)}, chunks={len(chunks)}, "
            f"chunk_size={self.chunk_context_size}"
        )
        
        return session_id
    
    def get_chunks(self, session_id: str, chunk_indices: List[int]) -> Dict[str, Any]:
        """
        Retrieve specific chunks (ONLY requested chunks).
        TRACKS which chunks have been retrieved by the agent.
        
        Args:
            session_id: Session identifier
            chunk_indices: List of chunk indices to retrieve
            
        Returns:
            Dict with requested chunk data and tracking info
        """
        if session_id not in self.storage:
            cache_operations_total.labels(operation="get", result="miss").inc()
            logger.warning(f"context_offload_miss - session_id={session_id}")
            return {
                "status": "error",
                "message": f"No data found for session_id: {session_id}"
            }
        
        stored = self.storage[session_id]
        total_chunks = stored["total_chunks"]
        
        # Validate chunk indices
        invalid_indices = [idx for idx in chunk_indices if idx < 0 or idx >= total_chunks]
        if invalid_indices:
            cache_operations_total.labels(operation="get", result="error").inc()
            return {
                "status": "error",
                "message": f"Invalid chunk indices: {invalid_indices}. Valid range: 0-{total_chunks-1}"
            }
        
        # Retrieve ONLY requested chunks
        chunks_data = []
        retrieval_time = datetime.utcnow().isoformat() + "Z"
        
        for idx in chunk_indices:
            chunks_data.extend(stored["chunks"][idx])
            
            # Update tracking for this chunk
            tracking = stored["chunk_tracking"][idx]
            tracking["retrieved"] = True
            tracking["retrieval_count"] += 1
            
            if tracking["first_retrieval"] is None:
                tracking["first_retrieval"] = retrieval_time
            tracking["last_retrieval"] = retrieval_time
        
        # Log retrieval operation
        stored["retrieval_history"].append({
            "timestamp": retrieval_time,
            "chunk_indices": chunk_indices,
            "rows_retrieved": len(chunks_data)
        })
        
        # Calculate remaining chunks
        remaining_indices = [
            i for i in range(total_chunks) 
            if not stored["chunk_tracking"][i]["retrieved"]
        ]
        
        # Calculate retrieval progress
        retrieved_count = sum(
            1 for t in stored["chunk_tracking"].values() 
            if t["retrieved"]
        )
        progress_percentage = (retrieved_count / total_chunks) * 100
        
        cache_operations_total.labels(operation="get", result="hit").inc()
        track_chunk_retrieval(session_id, len(chunk_indices))
        
        logger.info(
            f"context_offload_retrieved - session_id={session_id}, "
            f"chunks_requested={chunk_indices}, rows={len(chunks_data)}, "
            f"progress={progress_percentage:.1f}%, "
            f"retrieved={retrieved_count}/{total_chunks}"
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "retrieved_chunks": chunk_indices,
            "rows_in_response": len(chunks_data),
            "total_chunks_available": total_chunks,
            "chunks_remaining": len(remaining_indices),
            "remaining_chunk_indices": remaining_indices,
            "data": chunks_data,
            "retrieval_progress": {
                "chunks_retrieved": retrieved_count,
                "total_chunks": total_chunks,
                "percentage": round(progress_percentage, 1),
                "chunks_never_retrieved": remaining_indices
            },
            "message": (
                f"Retrieved {len(chunk_indices)} chunk(s) with {len(chunks_data)} rows. "
                f"Progress: {progress_percentage:.1f}% ({retrieved_count}/{total_chunks} chunks). "
                f"{len(remaining_indices)} chunk(s) remaining."
                if remaining_indices
                else f"All chunks retrieved! Total: {len(chunks_data)} rows across {total_chunks} chunks."
            )
        }
    
    def get_retrieval_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed retrieval statistics for a session"""
        if session_id not in self.storage:
            return None
        
        stored = self.storage[session_id]
        
        # Calculate stats
        total_chunks = stored["total_chunks"]
        retrieved_chunks = sum(1 for t in stored["chunk_tracking"].values() if t["retrieved"])
        never_retrieved = [
            idx for idx, t in stored["chunk_tracking"].items() 
            if not t["retrieved"]
        ]
        multiple_retrievals = [
            idx for idx, t in stored["chunk_tracking"].items() 
            if t["retrieval_count"] > 1
        ]
        
        return {
            "session_id": session_id,
            "total_chunks": total_chunks,
            "chunks_retrieved": retrieved_chunks,
            "chunks_never_retrieved": len(never_retrieved),
            "chunks_retrieved_multiple_times": len(multiple_retrievals),
            "progress_percentage": round((retrieved_chunks / total_chunks) * 100, 1),
            "never_retrieved_indices": never_retrieved,
            "multiple_retrieval_indices": multiple_retrievals,
            "retrieval_history": stored["retrieval_history"],
            "chunk_details": stored["chunk_tracking"]
        }
    
    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary without retrieving full data"""
        if session_id not in self.storage:
            return None
        
        return self.storage[session_id]["summary"]
    
    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about stored session including retrieval stats"""
        if session_id not in self.storage:
            return None
        
        stored = self.storage[session_id]
        
        # Add retrieval stats to metadata
        retrieved_count = sum(
            1 for t in stored["chunk_tracking"].values() 
            if t["retrieved"]
        )
        
        return {
            "session_id": session_id,
            "total_chunks": stored["total_chunks"],
            "total_rows": stored["total_rows"],
            "query": stored["query"],
            "year_filter": stored["year_filter"],
            "created_at": stored["created_at"],
            "retrieval_progress": {
                "chunks_retrieved": retrieved_count,
                "total_chunks": stored["total_chunks"],
                "percentage": round((retrieved_count / stored["total_chunks"]) * 100, 1)
            }
        }
    
    def clear(self, session_id: str) -> bool:
        """Clear stored session"""
        if session_id in self.storage:
            # Log final stats before clearing
            stats = self.get_retrieval_stats(session_id)
            logger.info(
                f"context_offload_cleared - session_id={session_id}, "
                f"retrieved={stats['chunks_retrieved']}/{stats['total_chunks']}, "
                f"never_retrieved={stats['chunks_never_retrieved']}"
            )
            
            del self.storage[session_id]
            cache_operations_total.labels(operation="delete", result="success").inc()
            return True
        return False


# Global instance
context_offloader = InMemoryContextOffloader(chunk_context_size=2000)