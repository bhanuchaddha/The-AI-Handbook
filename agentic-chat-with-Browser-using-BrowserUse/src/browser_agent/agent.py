import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller
from utils.helpers import sanitize_input

# Load environment variables
load_dotenv()

class BrowserAgent:
    """
    A class that implements browser agent functionality using the Browser Use library.
    """
    
    def __init__(self):
        """
        Initialize the browser agent with Browser Use components.
        """
        self.controller = Controller()
        
        # Configure browser to use the system's Chrome instance
        chrome_path = os.getenv("CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        
        try:
            # Set up browser configuration with only supported parameters
            # Use proper parameter name for Chrome path
            browser_config = BrowserConfig(
                chrome_instance_path=chrome_path
                # Removed user_data_dir as it's not a supported parameter
            )
            
            # Initialize browser with custom configuration
            self.browser = Browser(config=browser_config)
            
        except TypeError as e:
            # If chrome_instance_path isn't a valid parameter either, try without parameters
            print(f"Warning: BrowserConfig initialization error: {e}")
            print("Falling back to default browser configuration")
            self.browser = Browser(config=BrowserConfig())
        
        self.agent = None
        
    def __enter__(self):
        """
        Enter method for context management.
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit method for context management.
        """
        asyncio.run(self.close())
    
    def get_llm(self, provider="openai"):
        """
        Get the language model based on the specified provider.
        
        Args:
            provider (str): The LLM provider to use ('openai' or 'anthropic').
            
        Returns:
            The language model instance.
        """
        if provider == 'anthropic':
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.")
            
            return ChatAnthropic(
                model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0
            )
        elif provider == 'openai':
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Error: OPENAI_API_KEY is not set. Please provide a valid API key.")
            
            return ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        else:
            raise ValueError(f'Unsupported provider: {provider}')
        
    async def execute_instruction(self, instruction, provider="openai", max_steps=25):
        """
        Execute a browser instruction and return the result.
        
        Args:
            instruction (str): The instruction to execute.
            provider (str): The LLM provider to use.
            max_steps (int): Maximum number of steps for the agent to take.
            
        Returns:
            object: The history object from the agent execution.
        """
        # Sanitize user input for security
        clean_instruction = sanitize_input(instruction)
        
        try:
            # Initialize the LLM based on the provider
            llm = self.get_llm(provider)
            
            # Initialize the Agent with the configured browser
            self.agent = Agent(
                task=clean_instruction,
                llm=llm,
                controller=self.controller,
                browser=self.browser,
                use_vision=True,
                max_actions_per_step=1,
            )
            
            # Run the agent to complete the task and get history
            history = await self.agent.run(max_steps=max_steps)
            
            # Return the history object
            return history
            
        except Exception as e:
            # Return a string error message if exception occurs
            return f"Error executing browser instruction: {str(e)}\nPlease check your API keys and browser configuration and try again."
        
    async def close(self):
        """
        Close the browser instance.
        """
        try:
            await self.browser.close()
        except Exception as e:
            print(f"Error closing browser: {str(e)}")
