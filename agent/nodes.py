from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import AgentState
from langchain_core.runnables import RunnableConfig
from agent.llm import get_llm
from agent.chat_manager import chat_manager
from agent.prompts import SYSTEM_PROMPT_HELPDESK, DATABASE_SCHEMA
from database.client import run_query
from logs.log import logger
from langgraph.prebuilt import ToolNode
from agent.tools import TOOLS
from logs.log import logger

tool_node = ToolNode(TOOLS)

async def process_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Process user input and validate chat"""
    
    chat_id = state.get("chat_id")
    
    # Check token limit (from cache)
    if chat_id:
        within_limit = await chat_manager.check_token_limit(chat_id)
        
        if not within_limit:
            state["messages"].append(
                AIMessage(content="This chat has reached its maximum length. Please start a new chat to continue.")
            )
            return state
    
    logger.info(f"Input validated for chat {chat_id}")
    return state

async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate AI response using LLM with tool support"""
    
    llm = get_llm()
    messages = state["messages"]
    chat_id = state["chat_id"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Build message list
    full_messages = list(messages)
    
    # Add system message on first interaction
    if len([m for m in messages if isinstance(m, (HumanMessage, AIMessage))]) == 1:
        system_msg = SystemMessage(
            content= SYSTEM_PROMPT_HELPDESK + "\n\n" + DATABASE_SCHEMA
        )
        full_messages = [system_msg] + full_messages
    
    try:
        # Get user message
        user_msg_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg_content = msg.content
                break
        
        # Invoke LLM with tools
        logger.info(f"🤖 Generating response for chat {chat_id}")
        response = await llm_with_tools.ainvoke(full_messages, config=config)
        
        # Track tokens
        metadata = response.response_metadata.get("token_usage", {})
        input_tokens = metadata.get("prompt_tokens", 0) or metadata.get("input_tokens", 0)
        output_tokens = metadata.get("completion_tokens", 0) or metadata.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        print(f"\nInput: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}\n")
        
        # Add AI response to messages
        state["messages"].append(response)
        state["total_tokens"] = state.get("total_tokens", 0) + total_tokens
        
        # Cache user message if exists
        if user_msg_content:
            chat_manager.add_message_to_cache(
                chat_id=chat_id,
                role="user",
                content=user_msg_content,
                tokens=input_tokens
            )
        
        logger.info(f"✅ Response generated ({total_tokens} tokens)")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        state["messages"].append(
            AIMessage(content="I apologize, but I encountered an error. Please try again.")
        )
    
    return state

async def generate_response1(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate AI response using LLM"""
    
    llm = get_llm()
    messages = state["messages"]
    chat_id = state["chat_id"]
    
    # Build message list for LLM
    full_messages = list(messages)
    
    # Add system message if first interaction
    if len([m for m in messages if isinstance(m, (HumanMessage, AIMessage))]) == 1:
        system_msg = SystemMessage(
            content="You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
        )
        full_messages = [system_msg] + full_messages
    
    try:
        # Get user message (last HumanMessage)
        user_msg_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg_content = msg.content
                break
        
        # Invoke LLM
        logger.info(f"🤖 Generating response for chat {chat_id}")
        response = await llm.ainvoke(full_messages)
        metadata = response.response_metadata["token_usage"]

        input_tokens = metadata.get("prompt_tokens") or metadata.get("input_tokens")
        output_tokens = metadata.get("completion_tokens") or metadata.get("output_tokens")
        total_tokens = metadata["total_tokens"]
        print("\n\n")
        print("Input tokens:", input_tokens)
        print("Output tokens:", output_tokens)
        print("Total tokens:", total_tokens)
        print("\n\n")
        
        # Add AI response to state
        ai_message = AIMessage(content=response.content)
        state["messages"].append(ai_message)
        
        ai_msg_content = response.content
        
        estimated_user_tokens = int(input_tokens)
        estimated_ai_tokens = int(output_tokens)
        total_estimated = estimated_user_tokens + estimated_ai_tokens
        
        state["total_tokens"] = state.get("total_tokens", 0) + total_estimated
        
        # Add messages to CACHE ONLY (not DB yet)
        chat_manager.add_message_to_cache(
            chat_id=chat_id,
            role="user",
            content=user_msg_content,
            tokens=estimated_user_tokens
        )
        
        chat_manager.add_message_to_cache(
            chat_id=chat_id,
            role="assistant",
            content=ai_msg_content,
            tokens=estimated_ai_tokens
        )
        
        logger.info(f"✅ Response generated and cached for {chat_id} ({total_estimated} tokens)")
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}", exc_info=True)
        state["messages"].append(
            AIMessage(content="I apologize, but I encountered an error. Please try again.")
        )
    
    return state


async def save_messages(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    This node is now a NO-OP during normal flow.
    Messages are saved only when:
    1. User explicitly ends session
    2. User switches to another chat
    3. Session timeout occurs
    """
    
    chat_id = state.get("chat_id")
    logger.info(f"ℹ️ Messages remain cached for {chat_id} (will save on session end)")
    
    # Optional: You can add auto-save logic here if needed
    # For example, save every N messages or every X minutes
    
    return state


def should_continue(state: AgentState) -> str:
    """Determine if conversation should continue"""
    
    messages = state["messages"]
    
    # Check if last message indicates chat limit reached
    if messages and isinstance(messages[-1], AIMessage):
        if "reached its maximum length" in messages[-1].content:
            return "end"
    
    return "continue"

def route_after_llm(state: AgentState) -> str:
    """Route to tools if LLM made tool calls, otherwise continue"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"🔧 Routing to tool: {last_message.tool_calls[0]['name']}")
        return "tools"
    return "save"