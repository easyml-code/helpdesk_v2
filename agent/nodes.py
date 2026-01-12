from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from agent.state import AgentState
from langchain_core.runnables import RunnableConfig
from agent.llm import get_llm
from agent.chat_manager import chat_manager
from agent.prompts import SYSTEM_PROMPT_HELPDESK
from logs.log import logger, log_llm_call, set_trace_id, set_request_id
from metrics.prometheus import (
    track_llm_call, chat_messages_total, 
    chat_tokens_total, track_error
)
from langgraph.prebuilt import ToolNode
from agent.tools import TOOLS
import time


tool_node = ToolNode(TOOLS)


async def process_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Process user input and validate chat"""
    
    set_trace_id()
    set_request_id()
    
    chat_id = state.get("chat_id")
    
    # Check token limit using cumulative count
    if chat_id:
        within_limit = await chat_manager.check_token_limit(chat_id)
        
        if not within_limit:
            # Get token stats for logging
            stats = chat_manager.get_token_stats(chat_id)
            logger.warning(
                f"token_limit_reached - chat_id={chat_id}, "
                f"cumulative_total={stats.get('total', 0)}"
            )
            
            state["messages"].append(
                AIMessage(content="This chat has reached its maximum length. Please start a new chat to continue.")
            )
            track_error("token_limit_exceeded", "chat")
            return state
    
    logger.info(f"input_validated - chat_id={chat_id}")
    return state


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Generate AI response using LLM with tool support.
    FIXED: Properly tracks and accumulates tokens.
    """
    
    start_time = time.time()
    llm = get_llm()
    messages = state["messages"]
    chat_id = state["chat_id"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Build message list with system prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT_HELPDESK)
    
    # Apply windowing to prevent context overflow
    # Keep system + recent messages (windowing handled by checkpointer too)
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    full_messages = [system_msg] + list(recent_messages)
    
    logger.info(
        f"llm_invoke_start - chat_id={chat_id}, "
        f"input_messages={len(full_messages)}"
    )
    
    try:
        # Get user message content for caching
        user_msg_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg_content = msg.content
                break
        
        # Invoke LLM with tools
        response = await llm_with_tools.ainvoke(full_messages, config=config)
        
        # FIXED: Extract token usage properly
        metadata = response.response_metadata.get("token_usage", {})
        
        # Handle different response formats from different LLM providers
        input_tokens = (
            metadata.get("prompt_tokens") or 
            metadata.get("input_tokens") or 
            0
        )
        output_tokens = (
            metadata.get("completion_tokens") or 
            metadata.get("output_tokens") or 
            0
        )
        
        # If no token info, estimate (fallback)
        if input_tokens == 0 and output_tokens == 0:
            # Rough estimate: 4 chars per token
            input_tokens = sum(len(str(m.content)) for m in full_messages) // 4
            output_tokens = len(str(response.content)) // 4
            logger.warning(
                f"token_estimation_used - chat_id={chat_id}, "
                f"estimated_input={input_tokens}, estimated_output={output_tokens}"
            )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add AI response to state
        state["messages"].append(response)
        
        # Track metrics - FIXED: Use correct settings variable
        from config import settings
        track_llm_call(
            model=settings.LLM_MODEL,  # FIXED: Was LLM_MODEl
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=latency_ms / 1000,
            success=True
        )
        
        log_llm_call(
            model=settings.LLM_MODEL,  # FIXED: Was LLM_MODEl
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms
        )
        
        chat_messages_total.labels(role="assistant").inc()
        
        # FIXED: Properly cache with separate token counts
        if user_msg_content:
            # Cache user message with its input tokens
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="user",
                content=user_msg_content,
                tokens=input_tokens
            )
            
            # Cache assistant message with its output tokens
            ai_content = response.content if hasattr(response, 'content') else str(response)
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="assistant",
                content=ai_content,
                tokens=output_tokens
            )
            
            # Get updated stats
            stats_after = chat_manager.get_token_stats(chat_id)
            
            # Update state with cumulative total
            state["total_tokens"] = stats_after.get('total', 0)
            
            # Track in Prometheus
            chat_tokens_total.labels(chat_id=chat_id).inc(input_tokens + output_tokens)
        
        logger.info(
            f"llm_response_generated - chat_id={chat_id}, "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
            f"turn_total={input_tokens + output_tokens}, "
            f"cumulative_total={state.get('total_tokens', 0)}, "
            f"latency_ms={latency_ms:.2f}"
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"llm_error - chat_id={chat_id}, latency_ms={latency_ms:.2f}, error={e}",
            exc_info=True
        )
        
        track_llm_call(
            model="unknown",
            input_tokens=0,
            output_tokens=0,
            duration=latency_ms / 1000,
            success=False
        )
        track_error("llm_invocation_error", "agent")
        
        state["messages"].append(
            AIMessage(content="I apologize, but I encountered an error. Please try again.")
        )
    
    return state


async def save_messages(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Final save point - messages already cached during generation.
    Just logs completion.
    """
    
    chat_id = state.get("chat_id")
    
    # Get final token stats
    stats = chat_manager.get_token_stats(chat_id)
    
    logger.info(
        f"conversation_turn_complete - chat_id={chat_id}, "
        f"cumulative_total={stats.get('total', 0)}, "
        f"cumulative_input={stats.get('input', 0)}, "
        f"cumulative_output={stats.get('output', 0)}, "
        f"turns={len(stats.get('by_turn', []))}"
    )
    
    return state


def should_continue(state: AgentState) -> str:
    """Determine if conversation should continue"""
    
    messages = state["messages"]
    
    # Check if last message indicates limit reached
    if messages and isinstance(messages[-1], AIMessage):
        if "reached its maximum length" in messages[-1].content:
            return "end"
    
    return "continue"


def route_after_llm(state: AgentState) -> str:
    """Route to tools if LLM made tool calls, otherwise save"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
        logger.info(f"routing_to_tools - tools={tool_names}")
        chat_messages_total.labels(role="tool").inc()
        return "tools"
    
    return "save"