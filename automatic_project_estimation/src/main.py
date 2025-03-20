from model import ProjectPlan
import os
import yaml
from crewai import Agent, Task, Crew
from IPython.display import display, Markdown
import sys
from tabulate import tabulate


def load_config(filename: str) -> dict:
    """
    Load YAML configuration file relative to main script's location
    Args:
        filename: Name of the config file in the config directory
    Returns:
        dict: Configuration data
    """
    # Get the directory containing the main script
    main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    config_path = os.path.join(main_dir, "config", filename)
    
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        raise


def main(inputs):
    # Load configurations. configs are always stored in the config directory at same level as main file
    agents_config = load_config("agents.yaml")
    tasks_config = load_config("tasks.yaml")

    # Creating Agents
    project_planning_agent = Agent(config=agents_config["project_planning_agent"], cache=True)

    estimation_agent = Agent(config=agents_config["estimation_agent"], cache=True)

    resource_allocation_agent = Agent(config=agents_config["resource_allocation_agent"], cache=True)

    # Creating Tasks
    task_breakdown = Task(
        config=tasks_config["task_breakdown"], agent=project_planning_agent
    )

    time_resource_estimation = Task(
        config=tasks_config["time_resource_estimation"], agent=estimation_agent
    )

    resource_allocation = Task(
        config=tasks_config["resource_allocation"],
        agent=resource_allocation_agent,
        output_pydantic=ProjectPlan,  # This is the structured output we want
    )

    # Creating Crew
    crew = Crew(
        agents=[project_planning_agent, estimation_agent, resource_allocation_agent],
        tasks=[task_breakdown, time_resource_estimation, resource_allocation],
        verbose=True,
    )

    # Execute the Crew

    

    # Format the dictionary as Markdown for a better display in Jupyter Lab
    formatted_input = f"""
    **Project Type:** {inputs['project_type']}

    **Project Objectives:** {inputs['project_objectives']}

    **Industry:** {inputs['industry']}

    **Team Members:**
    {inputs['team_members']}
    **Project Requirements:**
    {inputs['project_requirements']}
    """
    # Display the formatted output as Markdown
    # display(Markdown(formatted_output))
    
    


    # Run the crew
    result = crew.kickoff(
    inputs=inputs
    )
 
    
    output = {
        'formatted_input': formatted_input,
        'usage_metrics': crew.usage_metrics,
        'result': result.pydantic,
        'tasks_output': result.tasks_output,
    }
    
    return output;



if __name__ == "__main__":
    
    
    # Define inputs for the App
    project = "Website"
    industry = "Technology"
    project_objectives = "Create a website for a small business"
    team_members = """
    - Michel John (Project Manager)
    - Susene Miraj (Frontend Engineer)
    - Boby Singh (Designer)
    - Alan Sorenson (Backend Engineer)
    - Travis Naveda (QA Engineer)
    """
    project_requirements = """
    - Create a responsive design that works well on desktop and mobile devices
    - Implement a modern, visually appealing user interface with a clean look
    - Develop a user-friendly navigation system with intuitive menu structure
    - Include an "About Us" page highlighting the company's history and values
    - Design a "Services" page showcasing the business's offerings with descriptions
    - Create a "Contact Us" page with a form and integrated map for communication
    - Implement a blog section for sharing industry news and company updates
    - Ensure fast loading times and optimize for search engines (SEO)
    - Integrate social media links and sharing capabilities
    - Include a testimonials section to showcase customer feedback and build trust
    """
    
    
    # Create Inputs dictionary
    inputs = {
    'project_type': project,
    'project_objectives': project_objectives,
    'industry': industry,
    'team_members': team_members,
    'project_requirements': project_requirements
    }
    
    # Run the main function with the inputs
    output = main(inputs)
    result = output['result']
    usage_metrics = output['usage_metrics']
    formatted_input = output['formatted_input']
    rasult_dict = result.dict()
    tasks = rasult_dict['tasks']
    milestones = rasult_dict['milestones']
    
    print("Usage Metrics")
    print(usage_metrics)
    print("Formatted Input")
    print(formatted_input)
    print("Tasks")
    print(tasks)
    print("Milestones")
    print(milestones)

    print("Tasks Output")
    print(output['tasks_output'])
    
    
    import pandas as pd
    df_tasks = pd.DataFrame(tasks)
    print("\nTask Details:")
    # Display the DataFrame as a table
    print(tabulate(df_tasks, headers='keys', tablefmt='grid'))



