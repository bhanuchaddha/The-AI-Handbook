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
        page_title="AI Operator using Browser-Use",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 AI Operator using Browser-Use")
    
    # Initialize and run chat interface
    chat_interface = ChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()
