import re
from typing import List, Dict, Any
import pandas as pd

__all__ = ['extract_sections', 'parse_task_details', 'parse_tasks_output', 'format_task_output', 'parse_crewai_output']

def extract_sections(markdown_text: str) -> Dict[str, str]:
    """Extract different sections from markdown text using headers."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in markdown_text.split('\n'):
        if line.startswith('###'):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('#', '').strip()
            current_content = []
        elif line.startswith('####'):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('#', '').strip()
            current_content = []
        else:
            current_content.append(line)
            
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
        
    return sections

def parse_task_details(markdown_text: str) -> pd.DataFrame:
    """Parse task details from markdown and convert to DataFrame."""
    tasks = []
    current_task = {}
    
    for line in markdown_text.split('\n'):
        if line.startswith('####'):
            if current_task:
                tasks.append(current_task)
            current_task = {'Task': line.replace('#', '').strip()}
        elif line.startswith('- '):
            key_value = line.replace('- ', '').split(': ', 1)
            if len(key_value) == 2:
                current_task[key_value[0]] = key_value[1]
                
    if current_task:
        tasks.append(current_task)
        
    return pd.DataFrame(tasks)

def parse_crewai_output(tasks_output: List[Any]) -> str:
    """Parse CrewAI task outputs into a formatted markdown string."""
    formatted_output = []
    
    for task in tasks_output:
        # Add agent header
        formatted_output.append(f"# Agent: {task.agent.strip()}")
        
        # Add task description
        if task.description:
            formatted_output.append("## Task:")
            formatted_output.append(task.description.strip())
            formatted_output.append("")
        
        # Add final answer/output
        if task.raw:
            formatted_output.append("## Final Answer:")
            formatted_output.append(task.raw.strip())
            formatted_output.append("")
            
        formatted_output.append("---")
        formatted_output.append("")
    
    return "\n".join(formatted_output)

def parse_tasks_output(tasks_output: List[Any]) -> Dict[str, Any]:
    """Parse the raw tasks output into a structured format."""
    parsed_output = {}
    
    for task in tasks_output:
        if task.raw:
            sections = extract_sections(task.raw)
            
            # Store the agent role
            parsed_output[f"Agent ({task.agent.strip()})"] = {
                "sections": sections,
                "summary": task.summary,
                "description": task.description,
                "raw": task.raw  # Store raw output for markdown display
            }
            
            # If there's a pydantic model, store its dict representation
            if task.pydantic:
                parsed_output[f"Agent ({task.agent.strip()})"]["structured_data"] = task.pydantic.dict()
    
    return parsed_output

def format_task_output(parsed_output: Dict[str, Any], format_type: str = 'markdown') -> str:
    """Format the parsed output into readable content."""
    # For raw markdown output, just concatenate the raw outputs
    if format_type == 'raw_markdown':
        formatted_content = []
        for agent, data in parsed_output.items():
            formatted_content.append(f"## {agent}")
            if 'raw' in data:
                formatted_content.append(data['raw'])
            formatted_content.append("---\n")
        return "\n\n".join(formatted_content)
    
    formatted_content = []
    
    for agent, data in parsed_output.items():
        formatted_content.append(f"## {agent}")
        
        if format_type == 'markdown':
            for section_name, content in data['sections'].items():
                formatted_content.append(f"### {section_name}")
                formatted_content.append(content)
                formatted_content.append("---")
        
        # Add structured data if available
        if 'structured_data' in data:
            formatted_content.append("### Structured Data")
            formatted_content.append("```json")
            formatted_content.append(str(data['structured_data']))
            formatted_content.append("```")
            
    return '\n\n'.join(formatted_content)
