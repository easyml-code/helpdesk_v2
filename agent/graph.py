from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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


def create_agent_graph():
    """Create the agent workflow graph"""
    
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
            "tools": "tools",      # If tool calls exist
            "save": "save_messages"  # If no tool calls
        }
    )
    
    # Permanent: tools → back to LLM
    workflow.add_edge("tools", "generate_response")
    
    # End after saving
    workflow.add_edge("save_messages", END)
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ Agent graph created with tool support")
    return app


# Global graph instance
agent_graph = create_agent_graph()