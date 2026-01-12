import streamlit as st
import requests
from datetime import datetime
import atexit
import time


# Configuration
API_BASE_URL = "http://localhost:8000/api"


def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = None
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_metrics" not in st.session_state:
        st.session_state.user_metrics = {}
    if "message_versions" not in st.session_state:
        st.session_state.message_versions = {}
    if "editing_message_id" not in st.session_state:
        st.session_state.editing_message_id = None


def handle_api_error(response):
    """Handle API errors including token expiration"""
    if response.status_code == 401:
        # Try to refresh token
        if try_refresh_token():
            return True
        else:
            # Token refresh failed, logout
            save_current_chat()
            st.session_state.clear()
            st.error("Session expired. Please login again.")
            st.rerun()
    return False


def try_refresh_token():
    """Try to refresh the access token"""
    if not st.session_state.refresh_token:
        return False
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            headers={
                "X-Refresh-Token": st.session_state.refresh_token
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data.get("refresh_token", st.session_state.refresh_token)
            return True
    except:
        pass
    
    return False


def save_current_chat():
    """Save current chat before switching or closing"""
    if st.session_state.current_chat_id and st.session_state.authenticated:
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/{st.session_state.current_chat_id}/end",
                headers={
                    "Authorization": f"Bearer {st.session_state.access_token}",
                    "X-Refresh-Token": st.session_state.refresh_token
                }
            )
            
            if response.status_code == 401:
                if not handle_api_error(response):
                    return
                # Retry after token refresh
                response = requests.post(
                    f"{API_BASE_URL}/chat/{st.session_state.current_chat_id}/end",
                    headers={
                        "Authorization": f"Bearer {st.session_state.access_token}",
                        "X-Refresh-Token": st.session_state.refresh_token
                    }
                )
            
            if response.status_code == 200:
                st.success("Chat saved successfully!")
            else:
                st.warning(f"Chat save returned status {response.status_code}")
        except Exception as e:
            st.error(f"Error saving chat: {str(e)}")


def login(email: str, password: str):
    """Authenticate user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            params={"email": email, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            st.session_state.authenticated = True
            return True
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False


def load_user_metrics():
    """Load user metrics"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics/user",
            headers={
                "Authorization": f"Bearer {st.session_state.access_token}",
                "X-Refresh-Token": st.session_state.refresh_token
            }
        )
        
        if response.status_code == 401:
            handle_api_error(response)
            return
        
        if response.status_code == 200:
            st.session_state.user_metrics = response.json()
    except Exception as e:
        st.warning(f"Could not load metrics: {str(e)}")


def load_chat_history():
    """Load all chats for the user"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/history",
            headers={
                "Authorization": f"Bearer {st.session_state.access_token}",
                "X-Refresh-Token": st.session_state.refresh_token
            }
        )
        
        if response.status_code == 401:
            handle_api_error(response)
            return
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.chat_history = data["chats"]
        else:
            st.error("Failed to load chat history")
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")


def load_chat_messages(chat_id: str):
    """Load messages for a specific chat"""
    try:
        # Save current chat first
        if st.session_state.current_chat_id and st.session_state.current_chat_id != chat_id:
            save_current_chat()
        
        response = requests.get(
            f"{API_BASE_URL}/chat/{chat_id}/messages",
            headers={
                "Authorization": f"Bearer {st.session_state.access_token}",
                "X-Refresh-Token": st.session_state.refresh_token
            }
        )
        
        if response.status_code == 401:
            handle_api_error(response)
            return
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.messages = [
                {
                    "role": msg["role"], 
                    "content": msg["content"],
                    "message_id": msg.get("message_id"),
                    "created_at": msg.get("created_at")
                }
                for msg in data["messages"]
            ]
            st.session_state.current_chat_id = chat_id
            st.success(f"Loaded chat: {chat_id[:16]}...")
        else:
            st.error("Failed to load messages")
    except Exception as e:
        st.error(f"Error loading messages: {str(e)}")


def send_message(message: str, topic: str = None):
    """Send message to chatbot"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "chat_id": st.session_state.current_chat_id,
                "topic": topic
            },
            headers={
                "Authorization": f"Bearer {st.session_state.access_token}",
                "X-Refresh-Token": st.session_state.refresh_token
            }
        )
        
        if response.status_code == 401:
            if not handle_api_error(response):
                return False
            # Retry
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "message": message,
                    "chat_id": st.session_state.current_chat_id,
                    "topic": topic
                },
                headers={
                    "Authorization": f"Bearer {st.session_state.access_token}",
                    "X-Refresh-Token": st.session_state.refresh_token
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.current_chat_id = data["chat_id"]
            
            # Update messages in session
            st.session_state.messages.append({
                "role": "user", 
                "content": message,
                "message_id": None,
                "created_at": datetime.utcnow().isoformat()
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": data["response"],
                "message_id": None,
                "created_at": datetime.utcnow().isoformat()
            })
            
            return True
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return False


def start_new_chat():
    """Start a new chat (saves old one first)"""
    # Save current chat
    if st.session_state.current_chat_id:
        save_current_chat()
    
    # Clear state for new chat
    st.session_state.current_chat_id = None
    st.session_state.messages = []
    st.success("New chat started!")


def logout():
    """Logout and save current chat"""
    save_current_chat()
    st.session_state.clear()
    st.success("Logged out successfully!")
    time.sleep(1)
    st.rerun()


def display_user_metrics_sidebar():
    """Display user metrics in sidebar"""
    if st.session_state.user_metrics:
        metrics = st.session_state.user_metrics
        
        st.sidebar.markdown("### 📊 Your Stats")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Chats", metrics.get("total_chats", 0))
            st.metric("Total Messages", metrics.get("total_messages", 0))
        
        with col2:
            st.metric("Total Turns", metrics.get("total_turns", 0))
            st.metric("Total Tokens", f"{metrics.get('total_tokens', 0):,}")
        
        avg_duration = metrics.get("avg_session_duration", 0)
        if avg_duration:
            st.sidebar.metric("Avg Session", f"{avg_duration:.1f}s")


def main():
    st.set_page_config(
        page_title="Helpdesk Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    # Save chat on app close (best effort)
    if st.session_state.authenticated and st.session_state.current_chat_id:
        atexit.register(save_current_chat)
    
    # Login screen
    if not st.session_state.authenticated:
        st.title("🤖 Helpdesk Agent Login")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if login(email, password):
                    st.success("Login successful!")
                    load_chat_history()
                    load_user_metrics()
                    st.rerun()
        return
    
    # Main chat interface
    st.title("🤖 Helpdesk Agent")
    
    # Sidebar - Chat History & Metrics
    with st.sidebar:
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄", use_container_width=True, help="Refresh"):
                load_chat_history()
                load_user_metrics()
                st.rerun()
        
        with col2:
            if st.button("➕", use_container_width=True, help="New Chat"):
                start_new_chat()
                load_chat_history()
                st.rerun()
        
        with col3:
            if st.button("💾", use_container_width=True, help="Save"):
                save_current_chat()
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            logout()
        
        st.divider()
        
        # User metrics
        display_user_metrics_sidebar()
        
        st.divider()
        
        # Chat history
        st.markdown("### 💬 Chat History")
        
        for chat in st.session_state.chat_history:
            chat_id = chat.get("chat_id")
            topic = chat.get("topic", "Untitled Chat")
            message_count = chat.get("message_count", 0)
            updated_at = chat.get("updated_at", "")
            
            # Format date
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                date_str = dt.strftime("%b %d, %H:%M")
            except:
                date_str = "Unknown"
            
            # Highlight current chat
            is_current = chat_id == st.session_state.current_chat_id
            button_type = "primary" if is_current else "secondary"
            
            # Chat button
            if st.button(
                f"{'📍 ' if is_current else ''}{topic[:20]}{'...' if len(topic) > 20 else ''}\n"
                f"📅 {date_str} | 💬 {message_count} msgs",
                key=chat_id,
                use_container_width=True,
                type=button_type
            ):
                if not is_current:
                    load_chat_messages(chat_id)
                    st.rerun()
    
    # Display current chat info
    if st.session_state.current_chat_id:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"💬 Chat ID: {st.session_state.current_chat_id[:20]}...")
        with col2:
            st.caption(f"📝 {len(st.session_state.messages)} messages")
    else:
        st.info("👋 Start a new conversation by typing a message below!")
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(content)
            elif role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(content)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Get topic if new chat (use first message as topic)
        topic = None
        if not st.session_state.current_chat_id:
            topic = user_input[:50]  # First 50 chars as topic
        
        # Send message
        with st.spinner("🤔 Thinking..."):
            if send_message(user_input, topic):
                load_chat_history()  # Refresh sidebar
                load_user_metrics()   # Refresh metrics
                st.rerun()


if __name__ == "__main__":
    main()