from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from langchain_core.runnables import RunnableConfig
from agent.chat_manager import chat_manager
from agent.prompts import get_llm_chain
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
    
    # Check token limit
    if chat_id:
        within_limit = await chat_manager.check_token_limit(chat_id)
        
        if not within_limit:
            stats = chat_manager.get_token_stats(chat_id)
            logger.warning(
                f"token_limit_reached - chat_id={chat_id}, "
                f"cumulative_total={stats.get('total', 0)}"
            )
            
            # Return error message WITHOUT adding to state
            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content="This chat has reached its maximum length. Please start a new chat.")
                ]
            }
    
    logger.info(f"input_validated - chat_id={chat_id}")
    return state


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate AI response using LLM with tool support"""
    
    start_time = time.time()
    messages = state["messages"]
    chat_id = state["chat_id"]
    
    # Use cached LLM chain (includes system prompt + tools)
    llm_chain = get_llm_chain()
    
    # NO WINDOWING HERE - AsyncWindowedCheckpointer handles it in graph
    # Just pass all messages as-is
    
    logger.info(
        f"llm_invoke_start - chat_id={chat_id}, total_messages={len(messages)}"
    )
    
    try:
        # Extract user message for caching
        user_msg_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg_content = msg.content
                break
        
        # Invoke LLM chain with all messages (checkpointer handles windowing)
        response = await llm_chain.ainvoke(messages, config=config)
        
        # Extract token usage
        metadata = response.response_metadata.get("token_usage", {})
        
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
        
        # Fallback estimation if no token info
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = sum(len(str(m.content)) for m in messages) // 4
            output_tokens = len(str(response.content)) // 4
            logger.warning(
                f"token_estimation_used - chat_id={chat_id}, "
                f"estimated_input={input_tokens}, estimated_output={output_tokens}"
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Track metrics in chat_manager
        chat_manager.track_llm_call(chat_id, latency_ms, success=True)
        
        # Track metrics in Prometheus
        from config import settings
        track_llm_call(
            model=settings.LLM_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=latency_ms / 1000,
            success=True
        )
        
        log_llm_call(
            model=settings.LLM_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms
        )
        
        chat_messages_total.labels(role="assistant").inc()
        ai_content = response.content if hasattr(response, 'content') else str(response)
        
        # Cache messages with proper token counts
        if isinstance(messages[-1], HumanMessage) and user_msg_content:
            # Cache user message with input tokens
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="user",
                content=user_msg_content,
                tokens=input_tokens
            )
            
            # Cache assistant message with output tokens
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="assistant",
                content=ai_content,
                tokens=output_tokens
            )
            
            # Update state with cumulative total
            stats_after = chat_manager.get_token_stats(chat_id)
            state["total_tokens"] = stats_after.get('total', 0)
            
            # Track in Prometheus
            chat_tokens_total.labels(chat_id=chat_id).inc(input_tokens + output_tokens)
        elif ai_content != "":
            # Cache assistant message with output tokens
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="assistant",
                content=ai_content,
                tokens=output_tokens
            )
            
            # Update state with cumulative total
            stats_after = chat_manager.get_token_stats(chat_id)
            state["total_tokens"] = stats_after.get('total', 0)
            
            # Track in Prometheus
            chat_tokens_total.labels(chat_id=chat_id).inc(output_tokens)

        logger.info(
            f"llm_response_generated - chat_id={chat_id}, "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
            f"turn_total={input_tokens + output_tokens}, "
            f"cumulative_total={state.get('total_tokens', 0)}, "
            f"latency_ms={latency_ms:.2f}"
        )
        
        # Return only the NEW message (LangGraph will add it to state)
        return {"messages": [response]}
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"llm_error - chat_id={chat_id}, latency_ms={latency_ms:.2f}, error={e}",
            exc_info=True
        )
        
        # Track error
        chat_manager.track_llm_call(chat_id, latency_ms, success=False)
        
        track_llm_call(
            model="unknown",
            input_tokens=0,
            output_tokens=0,
            duration=latency_ms / 1000,
            success=False
        )
        track_error("llm_invocation_error", "agent")
        
        return {
            "messages": [
                AIMessage(content="I apologize, but I encountered an error. Please try again.")
            ]
        }


async def save_messages(state: AgentState, config: RunnableConfig) -> AgentState:
    """Final save point - messages cached during generation"""
    
    chat_id = state.get("chat_id")
    stats = chat_manager.get_token_stats(chat_id)
    
    logger.info(
        f"conversation_turn_complete - chat_id={chat_id}, "
        f"cumulative_total={stats.get('total', 0)}, "
        f"cumulative_input={stats.get('input', 0)}, "
        f"cumulative_output={stats.get('output', 0)}, "
        f"turns={len(stats.get('by_turn', []))}"
    )
    
    return state


def route_after_llm(state: AgentState) -> str:
    """Route to tools if LLM made tool calls, otherwise save"""
    last_message = state["messages"][-1]
    chat_id = state["chat_id"]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
        logger.info(f"routing_to_tools - tools={tool_names}")
        chat_messages_total.labels(role="tool").inc()
        
        # Track tool execution
        chat_manager.track_tool_execution(chat_id)
        
        return "tools"
    
    return "save"