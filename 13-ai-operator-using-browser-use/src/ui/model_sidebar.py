import streamlit as st
import os
from src.utils.model_config import MODEL_PROVIDERS

class ModelSidebar:
    """
    A reusable sidebar component for selecting LLM providers and models in Streamlit apps.
    """
    def __init__(self):
        if "provider" not in st.session_state:
            st.session_state.provider = list(MODEL_PROVIDERS.keys())[0]
        if "model" not in st.session_state:
            st.session_state.model = MODEL_PROVIDERS[st.session_state.provider]["models"][0]
        
    def render(self):
        """
        Renders the sidebar for model selection and returns the selected configuration.
        """
        with st.sidebar:
            st.title("Model Settings")
            
            # Provider selection
            provider = st.selectbox(
                "Select Provider", 
                options=list(MODEL_PROVIDERS.keys()),
                key="provider"
            )
            
            # Model selection based on provider
            provider_config = MODEL_PROVIDERS[provider]
            model = st.selectbox(
                "Select Model", 
                options=provider_config["models"],
                key="model"
            )
            
            # API key input (if needed)
            api_key = None
            if provider_config["api_key_env"]:
                env_key = provider_config["api_key_env"]
                api_key = st.text_input(
                    f"{provider} API Key", 
                    type="password",
                    value=os.environ.get(env_key, ""),
                    help=f"Enter your {provider} API key",
                    key=f"{provider.lower()}_api_key"
                )
                if api_key:
                    os.environ[env_key] = api_key
                
            # Show warning for Ollama
            if provider == "Ollama" and "warning" in provider_config:
                st.warning(provider_config["warning"])
            
            st.divider()
                
        return {
            "provider": provider.lower(),
            "model": model,
            "api_key": api_key
        }
