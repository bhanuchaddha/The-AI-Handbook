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
    logging.info("Initializing new session")

if "rag_tool" not in st.session_state:
    st.session_state.rag_tool = RagTool()

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(uploaded_file):
    if uploaded_file is not None:
        # Display PDF preview in the sidebar
        st.header("PDF Preview")
        pdf_bytes = uploaded_file.getvalue()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type=["pdf"])

    if uploaded_file:
        file_name = uploaded_file.name
        logging.info(f"Processing uploaded file: {file_name}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                status_placeholder = st.empty()
                status_placeholder.info(f"Processing document: {file_name}")
                
                query_engine = st.session_state.rag_tool.process_document(temp_dir)
                st.session_state.current_query_engine = query_engine

                # Status will be logged by RagTool
                st.success("Ready to Chat!")
                
        except Exception as e:
            logging.exception("Error processing document")
            st.error(f"An error occurred: {e}")
            st.stop()

# Chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.header("RAG over PDF API")
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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # # Simulate stream of response with milliseconds delay
        # streaming_response = query_engine.query(prompt)
        
        # for chunk in streaming_response.response_gen:
        #     full_response += chunk
        #     message_placeholder.markdown(full_response + "▌")

        full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
