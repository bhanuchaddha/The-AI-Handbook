import yaml
import os
from crewai import Agent, Task, Crew
from langchain_experimental.tools import PythonREPLTool
from typing import Dict

class APICrewHandler:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load YAML configuration files."""
        base_path = os.path.join(os.path.dirname(__file__), '..', 'config')
        
        with open(os.path.join(base_path, 'agents.yaml'), 'r') as f:
            agents_config = yaml.safe_load(f)
        
        with open(os.path.join(base_path, 'tasks.yaml'), 'r') as f:
            tasks_config = yaml.safe_load(f)
            
        return {'agents': agents_config, 'tasks': tasks_config}
    
    def _create_rag_tool(self):
        """Create a tool for RAG queries."""
        def rag_search(query: str) -> str:
            response = self.query_engine.query(query)
            return str(response)
            
        return Tool(
            name="rag_search",
            func=rag_search,
            description="Search through API documentation using RAG"
        )

    def process_query(self, user_query: str, auth_token: str = None):
        """Process user query using CrewAI."""
        # Create agents
        analyst = Agent(
            role=self.config['agents']['api_analyst']['role'],
            goal=self.config['agents']['api_analyst']['goal'],
            backstory=self.config['agents']['api_analyst']['backstory'],
            tools=[self._create_rag_tool()],
            verbose=True
        )
        
        developer = Agent(
            role=self.config['agents']['python_developer']['role'],
            goal=self.config['agents']['python_developer']['goal'],
            backstory=self.config['agents']['python_developer']['backstory'],
            tools=[PythonREPLTool()],
            verbose=True
        )
        
        # Create tasks
        analyze_task = Task(
            description=self.config['tasks']['analyze_documentation']['description'].format(
                user_query=user_query
            ),
            expected_output=self.config['tasks']['analyze_documentation']['expected_output'],
            agent=analyst
        )
        
        implement_task = Task(
            description=self.config['tasks']['implement_api_call']['description'].format(
                api_details="{analyze_task_output}",
                auth_token=auth_token or "None"
            ),
            expected_output=self.config['tasks']['implement_api_call']['expected_output'],
            agent=developer
        )
        
        # Create and run crew
        crew = Crew(
            agents=[analyst, developer],
            tasks=[analyze_task, implement_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result
