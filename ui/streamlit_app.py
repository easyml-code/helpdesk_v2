import streamlit as st
import requests
from datetime import datetime
import atexit


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


def save_current_chat():
    """Save current chat before switching or closing"""
    if st.session_state.current_chat_id:
        try:
            requests.post(
                f"{API_BASE_URL}/chat/{st.session_state.current_chat_id}/end",
                headers={
                    "Authorization": f"Bearer {st.session_state.access_token}",
                    "X-Refresh-Token": st.session_state.refresh_token
                }
            )
            st.success("Chat saved successfully!")
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
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.messages = [
                {"role": msg["role"], "content": msg["content"]}
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
            # Session expired
            save_current_chat()
            st.session_state.clear()
            st.error("Session expired. Please login again.")
            st.rerun()
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.current_chat_id = data["chat_id"]
            
            # Update messages in session
            st.session_state.messages.append({"role": "user", "content": message})
            st.session_state.messages.append({"role": "assistant", "content": data["response"]})
            
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


def main():
    st.set_page_config(
        page_title="Helpdesk Agent",
        page_icon="",
        layout="wide"
    )
    
    init_session_state()
    
    # Save chat on app close (best effort)
    if st.session_state.authenticated and st.session_state.current_chat_id:
        atexit.register(save_current_chat)
    
    # Login screen
    if not st.session_state.authenticated:
        st.title("Login")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if login(email, password):
                    st.success("Login successful!")
                    load_chat_history()
                    st.rerun()
        return
    
    # Main chat interface
    st.title("Helpdesk Agent")
    
    # Sidebar - Chat History
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                load_chat_history()
                st.rerun()
        
        with col2:
            if st.button("New Chat", use_container_width=True):
                start_new_chat()
                load_chat_history()
                st.rerun()
        
        if st.button("Save Chat", use_container_width=True):
            save_current_chat()
        
        if st.button("Logout", use_container_width=True):
            save_current_chat()
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Display chat history
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
                f"{topic[:25]}{'...' if len(topic) > 25 else ''}\n"
                f"{date_str} | {message_count} msgs",
                key=chat_id,
                use_container_width=True,
                type=button_type
            ):
                if not is_current:
                    load_chat_messages(chat_id)
                    st.rerun()
    
    # Display current chat info
    if st.session_state.current_chat_id:
        st.caption(f"Chat ID: {st.session_state.current_chat_id[:20]}... | {len(st.session_state.messages)} messages")
    else:
        st.info("How may I assist you today?")
    
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
        with st.spinner("Thinking..."):
            if send_message(user_input, topic):
                load_chat_history()  # Refresh sidebar
                st.rerun()


if __name__ == "__main__":
    main()