import streamlit as st
import sys
import os
import asyncio
from dotenv import load_dotenv

# Add the project root to the Python path to fix import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.ui.chat_interface import ChatInterface
from src.browser_agent.agent import BrowserAgent

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Agentic Chat with Browser",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Agentic Chat with Browser")
    
    # Display API key inputs
    with st.sidebar:
        st.header("Configuration")
        provider = st.radio("Select LLM Provider:", ["openai", "anthropic"], index=0)
        
        if provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                openai_api_key = st.text_input("OpenAI API Key", type="password")
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
        elif provider == "anthropic":
            if "ANTHROPIC_API_KEY" not in os.environ:
                anthropic_api_key = st.text_input("Anthropic API Key", type="password")
                if anthropic_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    
    # Initialize browser agent
    browser_agent = BrowserAgent()
    
    # Initialize chat interface
    chat_interface = ChatInterface(browser_agent, provider)
    
    # Run the chat interface
    chat_interface.run()

if __name__ == "__main__":
    main()
