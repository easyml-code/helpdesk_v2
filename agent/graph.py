from langgraph.graph import StateGraph, END
from agent.checkpointer import AsyncWindowedCheckpointer, AsyncAdaptiveWindowedCheckpointer
from agent.state import AgentState
from agent.nodes import (
    process_input, 
    generate_response, 
    save_messages,
    route_after_llm,
    tool_node
)
from logs.log import logger
from config import settings


def create_agent_graph(use_adaptive: bool = True, window_size: int = 10):
    """
    Create agent workflow graph with async windowed checkpointing
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("tools", tool_node)
    workflow.add_node("save_messages", save_messages)
    
    # Entry point
    workflow.set_entry_point("process_input")
    
    # Edges
    workflow.add_edge("process_input", "generate_response")
    
    # Conditional: LLM → tools OR save
    workflow.add_conditional_edges(
        "generate_response",
        route_after_llm,
        {
            "tools": "tools",
            "save": "save_messages"
        }
    )
    
    # tools → back to LLM
    workflow.add_edge("tools", "generate_response")
    
    # End after saving
    workflow.add_edge("save_messages", END)
    
    # Create async checkpointer
    if use_adaptive:
        checkpointer = AsyncAdaptiveWindowedCheckpointer(
            base_window_size=window_size,
            max_window_tokens=getattr(settings, 'MAX_WINDOW_TOKENS', 8000),
            min_window_size=max(4, window_size // 2)
        )
        logger.info(
            f"agent_graph_created - checkpointer=async_adaptive, "
            f"window_size={window_size}, max_tokens={settings.LLM_MAX_TOKENS}"
        )
    else:
        checkpointer = AsyncWindowedCheckpointer(window_size=window_size)
        logger.info(
            f"agent_graph_created - checkpointer=async_windowed, window_size={window_size}"
        )
    
    # Compile with async checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("agent_graph_compiled_with_async_checkpointing")
    return app


# Global graph instance
agent_graph = create_agent_graph(
    use_adaptive=True,
    window_size=getattr(settings, 'CHECKPOINT_WINDOW_SIZE', 10)
)