from langgraph.graph import StateGraph, END
from agent.checkpointer import WindowedCheckpointer, AdaptiveWindowedCheckpointer
from langchain_core.runnables import RunnableConfig
from agent.state import AgentState
from agent.nodes import (
    process_input, 
    generate_response, 
    save_messages,
    should_continue,
    route_after_llm,
    tool_node
)
from logs.log import logger
from config import settings


def create_agent_graph(use_adaptive: bool = True, window_size: int = 10):
    """
    Create the agent workflow graph with windowed checkpointing
    
    Args:
        use_adaptive: Use adaptive windowing based on tokens (default: True)
        window_size: Base window size for messages (default: 10)
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
    
    # Permanent: tools → back to LLM
    workflow.add_edge("tools", "generate_response")
    
    # End after saving
    workflow.add_edge("save_messages", END)
    
    # Create checkpointer based on configuration
    if use_adaptive:
        checkpointer = AdaptiveWindowedCheckpointer(
            base_window_size=window_size,
            max_window_tokens=getattr(settings, 'MAX_WINDOW_TOKENS', 8000),
            min_window_size=max(4, window_size // 2)
        )
        logger.info(
            f"agent_graph_created - checkpointer=adaptive, "
            f"window_size={window_size}, max_tokens={settings.LLM_MAX_TOKENS}"
        )
    else:
        checkpointer = WindowedCheckpointer(window_size=window_size)
        logger.info(
            f"agent_graph_created - checkpointer=windowed, window_size={window_size}"
        )
    
    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("agent_graph_compiled_with_tools_and_windowing")
    return app


# Global graph instance with default settings
agent_graph = create_agent_graph(
    use_adaptive=True,
    window_size=getattr(settings, 'CHECKPOINT_WINDOW_SIZE', 10)
)