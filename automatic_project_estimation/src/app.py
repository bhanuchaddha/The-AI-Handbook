import streamlit as st
import os
from main import main
import pandas as pd
from tabulate import tabulate
import sys
from pathlib import Path
import base64

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils import parse_tasks_output, parse_crewai_output

# Function to load and encode image
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Page config
st.set_page_config(
    page_title="Automatic Project Estimation",
    page_icon="📊",
    layout="wide"
)

# Load CrewAI logo
crew_ai_logo = get_img_as_base64(os.path.join(project_root, "assets", "crew_ai_logo.png"))

# Main title with logo
if crew_ai_logo:
    st.markdown("""
        # Automatic Project Estimation with <img src="data:image/png;base64,{}" width="150" style="vertical-align: top;">
    """.format(crew_ai_logo), unsafe_allow_html=True)
else:
    st.title("Automatic Project Estimation with CrewAI")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully!")



# Input form
with st.form("project_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        project_type = st.text_input("Project Type", "Website")
        industry = st.text_input("Industry", "Technology")
        
    with col2:
        project_objectives = st.text_area("Project Objectives", "Create a website for a small business")
    
    team_members = st.text_area("Team Members", """
    - Michel John (Project Manager)
    - Susene Miraj (Frontend Engineer)
    - Boby Singh (Designer)
    - Alan Sorenson (Backend Engineer)
    - Travis Naveda (QA Engineer)
    """)
    
    project_requirements = st.text_area("Project Requirements", """
    - Create a responsive design that works well on desktop and mobile devices
    - Implement a modern, visually appealing user interface with a clean look
    - Develop a user-friendly navigation system with intuitive menu structure
    - Include an "About Us" page highlighting the company's history and values
    - Design a "Services" page showcasing the business's offerings with descriptions
    """)
    
    submitted = st.form_submit_button("Generate Estimation")

if submitted:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar first!")
    else:
        with st.spinner('Generating project estimation...'):
            # Prepare inputs
            inputs = {
                'project_type': project_type,
                'project_objectives': project_objectives,
                'industry': industry,
                'team_members': team_members,
                'project_requirements': project_requirements
            }

            
            # Call main function
            output = main(inputs)

            # Parse and display task outputs
            parsed_outputs = parse_tasks_output(output['tasks_output'])
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Project Overview", 
                "Milestones", 
                "Tasks", 
                "Usage Metrics",
                "Agent's Chain of Thought"
            ])
            
            with tab1:
                st.markdown(output['formatted_input'])
                
            with tab2:
                st.markdown("### Milestones")
                result_dict = output['result'].dict()
                milestones_df = pd.DataFrame(result_dict['milestones'])
                st.dataframe(milestones_df, use_container_width=True)
            
            with tab3:
                st.markdown("### Tasks")
                result_dict = output['result'].dict()
                tasks_df = pd.DataFrame(result_dict['tasks'])
                st.dataframe(tasks_df, use_container_width=True)
            
            with tab4:
                st.markdown("### Usage Metrics")
                metrics = output['usage_metrics']
                st.json(metrics.__dict__)
                
                # Calculate and display costs
                million_tokens_used = (metrics.prompt_tokens + metrics.completion_tokens) / 1_000_000
                costs = 0.150 * million_tokens_used
                st.metric("Estimated Cost", f"${costs:.4f}")
            
            with tab5:
                st.markdown("### Agent's Chain of Thought")
                # Display CrewAI logs in markdown format
                crewai_logs = parse_crewai_output(output['tasks_output'])
                st.markdown(crewai_logs)

# Add footer
st.markdown("---")
st.markdown("Built by [Bhanu Chaddha](https://www.linkedin.com/in/bhanu-chaddha/) with ❤️")
