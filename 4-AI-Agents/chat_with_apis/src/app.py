import os
import gc
import tempfile
import uuid
import base64
import pandas as pd
import streamlit as st
from io import BytesIO
from tools.rag_tool import RagTool
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.rag_tool = RagTool()
    try:
        st.session_state.current_query_engine = st.session_state.rag_tool.get_query_engine()
        st.success("Loaded existing documents from knowledge base!")
    except Exception as e:
        st.session_state.current_query_engine = None
        st.info("No existing documents found. Please upload some documents to get started.")
    logging.info("Initializing new session")

if "rag_tool" not in st.session_state:
    st.session_state.rag_tool = RagTool()

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    # Don't reset query engine on chat reset
    gc.collect()

# Define supported document types (all types that Docling supports natively)
SUPPORTED_DOCUMENT_TYPES = ["pdf", "docx", "doc", "html", "htm", "epub", "md", "txt", "rtf", "odt"]

# Sidebar for file upload
with st.sidebar:
    st.header("Document Chat")
    st.write("Upload any document in these formats: " + ", ".join(SUPPORTED_DOCUMENT_TYPES))
    uploaded_file = st.file_uploader(
        "Choose documents", 
        type=SUPPORTED_DOCUMENT_TYPES,
        accept_multiple_files=True,  # Enable multiple file upload
        help="Upload one or more documents in supported formats"
    )

    for file in uploaded_file:
        file_name = file.name
        file_type = file_name.split('.')[-1].lower()
        logging.info(f"Processing uploaded file: {file_name} (type: {file_type})")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                status_placeholder = st.empty()
                status_placeholder.info(f"Processing {file_type.upper()} document: {file_name}")
                
                query_engine = st.session_state.rag_tool.process_document(temp_dir)
                st.session_state.current_query_engine = query_engine
                st.success(f"Added {file_name} to knowledge base!")
                
        except Exception as e:
            logging.exception(f"Error processing {file_type} document")
            st.error(f"Error processing {file_name}: {str(e)}")
            continue

# Chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Chat with your Documents")  # Updated header
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_tool.query_with_debug(prompt)
                
                # Show retrieved chunks and thinking process
                with st.expander("🔍 Search and Reasoning", expanded=False):
                    st.markdown("### Retrieved Context")
                    debug_info = st.session_state.rag_tool.get_query_debug_info()
                    if debug_info and debug_info["nodes_by_relevance"]:
                        for idx, node in enumerate(debug_info["nodes_by_relevance"]):
                            st.markdown("---")
                            st.markdown(f"**Source**: {node.metadata.get('file_name', 'unknown')}")
                            st.markdown(f"**Score**: {node.score:.4f}")
                            st.markdown(f"```\n{node.text}\n```")
                    
                    # Show thinking process if available
                    if hasattr(response, 'metadata') and response.metadata:
                        st.markdown("### LLM Reasoning")
                        if 'prompt' in response.metadata:
                            st.markdown("**System Prompt:**")
                            st.markdown(f"```\n{response.metadata['prompt']}\n```")
                        if 'context' in response.metadata:
                            st.markdown("**Context Analysis:**")
                            st.markdown(f"```\n{response.metadata['context']}\n```")
                
                # Show final response
                message_placeholder.markdown(str(response))
        
        except Exception as e:
            logging.error(f"Query error: {e}")
            message_placeholder.error("Error processing query. Please try again.")
            raise e
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
