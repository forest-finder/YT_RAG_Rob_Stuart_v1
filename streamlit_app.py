#!/usr/bin/env python3
"""
Streamlit Web Interface for Terminal RAG System
A web-based chat interface for your YouTube video RAG system.
"""

import streamlit as st
from terminal_rag import TerminalRAGSystem
import time

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="🎥",
    layout="wide"
)

# Initialize the RAG system
@st.cache_resource
def init_rag_system():
    """Initialize and cache the RAG system."""
    # Override environment variables with Streamlit secrets if available
    import os
    if hasattr(st, 'secrets'):
        try:
            if 'OPENAI_API_KEY' in st.secrets:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
            if 'SUPABASE_URL' in st.secrets:
                os.environ['SUPABASE_URL'] = st.secrets['SUPABASE_URL']
            if 'SUPABASE_SERVICE_KEY' in st.secrets:
                os.environ['SUPABASE_SERVICE_KEY'] = st.secrets['SUPABASE_SERVICE_KEY']
        except Exception as e:
            st.warning(f"Could not load secrets: {e}")
    
    return TerminalRAGSystem(max_history=10)

# App title and description
st.title("🎥 YouTube RAG Chat System")
st.markdown("Ask questions about your YouTube video content!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    try:
        st.session_state.rag_system = init_rag_system()
        st.success("✅ Connected to database and AI models!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.stop()

# Sidebar with controls
with st.sidebar:
    st.header("🛠️ Controls")
    
    # Clear conversation button
    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.rag_system.clear_conversation()
        st.rerun()
    
    # Show conversation history
    if st.button("📝 Show History"):
        if st.session_state.rag_system.conversation_history:
            st.write("**Conversation History:**")
            for msg in st.session_state.rag_system.conversation_history:
                role = "You" if msg.role == "user" else "Assistant"
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                st.write(f"**[{timestamp}] {role}:** {msg.content[:100]}...")
        else:
            st.write("No conversation history yet.")
    
    # Settings
    st.header("⚙️ Settings")
    st.write(f"**Similarity Threshold:** {st.session_state.rag_system.similarity_threshold}")
    st.write(f"**Max Results:** {st.session_state.rag_system.max_results}")
    st.write(f"**Model:** {st.session_state.rag_system.chat_model}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your YouTube videos..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching videos and generating response..."):
            try:
                # Process the query using your existing RAG system
                response = st.session_state.rag_system.process_query(prompt)
                
                if response:
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.markdown("❌ No response generated. Please try a different question.")
            
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Ask questions about your video content, use 'search [query]' for search-only results")