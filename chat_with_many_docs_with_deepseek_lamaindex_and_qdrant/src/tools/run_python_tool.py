from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.llms import Ollama  # Changed import to Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os



if __name__ == "__main__":
    # Removed OpenAI API key setup

    # 1. Initialize Language Model and Prompt
    llm = Ollama(model="codellama:7b")  # Initialize Ollama with codellama:7b
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Generate Python code for {query}. Please provide only the raw Python code without any additional formatting, comments, or markers like ``` or [/PYTHON] . Return only code without comments or explaintons.Dont add any format indicatory. dont add ``` or ```python or [/PYTHON] or [PYTHON]",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # 2. Initialize the PythonREPLTool
    python_repl = PythonREPLTool()


    # 3. Prompt for code generation & execution
    user_prompt = "execute GET method on https://jsonplaceholder.typicode.com/posts/1 and return the response body"
    generated_code = chain.run(user_prompt)
    print("Generated Code:")
    print(generated_code)

    # 4. Remove ```python to prevent errors when running.
    generated_code = generated_code.replace("```python", "").replace("```", "")
    generated_code = generated_code.replace("``````python", "").replace("``````", "") # Added to handle different code block formats
    generated_code = generated_code.replace("```", "") # Added to handle different code block formats


    # 5. Execute generated code
    execution_result = python_repl.run(generated_code)


    # 6. Print the result
    print("Generated Code:")
    print(generated_code)
    print("\nExecution Result:")
    print(execution_result)
    