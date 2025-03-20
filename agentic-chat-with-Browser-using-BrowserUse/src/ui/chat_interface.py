import streamlit as st
import asyncio
import re
from src.browser_agent.agent import BrowserAgent

class ChatInterface:
    """
    A class that implements the Streamlit chat interface for the browser agent.
    """
    
    def __init__(self, browser_agent, provider="openai"):
        """
        Initialize the chat interface with a browser agent.
        
        Args:
            browser_agent (BrowserAgent): The browser agent to use.
            provider (str): The LLM provider to use.
        """
        self.browser_agent = browser_agent
        self.provider = provider
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def run(self):
        """
        Run the chat interface.
        """
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like the browser to do?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Executing browser instructions..."):
                    response_placeholder = st.empty()
                    
                    # Define the async function to execute the instruction
                    async def execute_async_instruction():
                        result = await self.browser_agent.execute_instruction(prompt, provider=self.provider)
                        
                        # Check if result is a string (error or legacy response)
                        if isinstance(result, str):
                            return result
                        
                        try:
                            # Get the final result if available
                            if hasattr(result, 'final_result') and result.final_result():
                                return result.final_result()
                            
                            # Otherwise look for extracted content
                            if hasattr(result, 'extracted_content'):
                                contents = result.extracted_content()
                                if contents:
                                    return contents[-1]  # Return the last extracted content
                            
                            # If still nothing useful, try to get any result message
                            if hasattr(result, 'action_results'):
                                action_results = result.action_results()
                                for res in reversed(action_results):  # Check results in reverse order
                                    if hasattr(res, 'extracted_content') and res.extracted_content:
                                        return res.extracted_content
                            
                            # Fallback to a basic completion message
                            if hasattr(result, 'is_successful'):
                                success = result.is_successful()
                                if success is True:
                                    return "Task completed successfully."
                                elif success is False:
                                    return "Task failed to complete."
                            
                            # If we can't access any history methods, return a generic message
                            return "Task executed but no detailed results available."
                            
                        except Exception as e:
                            # If any error occurs processing the history object
                            return f"Task executed but error processing results: {str(e)}"
                    
                    # Run the async function
                    response = asyncio.run(execute_async_instruction())
                    response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
