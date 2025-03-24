# AI Operator Using Browser-Use

[![YouTube Video](https://img.youtube.com/vi/y19HLcPLbKc/0.jpg)](https://www.youtube.com/watch?v=y19HLcPLbKc)


This project is a Python application that utilizes Streamlit as a user interface to send chat instructions to an AI browser agent using the Browser Use library. It allows users to interact with AI-powered browser automation capabilities and receive responses in real-time.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **AI Browser Automation**: Use natural language to instruct the AI to perform web tasks.
- **Real-time Chat Interface**: User-friendly Streamlit UI for conversational interactions.
- **Multiple LLM Support**: Integration with various language models including OpenAI and Anthropic.
- **Model Selection**: Choose different language models through the UI sidebar.
- **Web Automation**: Leverage Playwright for browser control and automation.
- **Visual Feedback**: View browser interaction results directly in the chat interface.
- **Extensibility**: Modular design allows for easy addition of new features and models.

## Prerequisites

Before installing the project, you need:

- Python 3.12 or higher
- A modern web browser
- API keys for at least one LLM provider (OpenAI or Anthropic)
- Sufficient permissions to install browser drivers automatically

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/bhanuchaddha/The-AI-Handbook.git
   cd The-AI-Handbook/13-ai-operator-using-browser-use
   ```

2. Create a virtual environment (highly recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Playwright browsers:
   ```
   playwright install
   ```

## Configuration

1. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   # Add other model API keys as needed
   ```

## Usage

To run the application, execute the following command:
```
streamlit run src/main.py
```

This will start the Streamlit server and open the application in your default web browser. You can then:

1. Select your preferred language model in the sidebar.
2. Type instructions in the chat interface to direct the AI browser agent.
3. View the results of browser interactions and agent responses.
4. Continue the conversation with follow-up questions or new tasks.

Example interactions:
- "Search for the latest AI research papers on arXiv."
- "Go to weather.gov and tell me the forecast for San Francisco."
- "Find the top 5 tech news headlines on TechCrunch."

## Project Architecture

The application follows a modular architecture:

1. **Main Application**: Entry point that initializes the Streamlit interface.
2. **Browser Agent**: Core component that handles browser automation using the Browser Use library.
3. **Chat Interface**: Manages user interactions, message history, and display of results.
4. **Model Configuration**: Handles the loading and configuration of language models.

The data flow works as follows:
1. User inputs a message through the Streamlit interface.
2. The message is processed by the selected language model.
3. Browser instructions are interpreted and executed by the Browser Use agent.
4. Results are formatted and displayed back to the user in the chat interface.

## Project Structure

```
13-ai-operator-using-browser-use
├── src
│   ├── main.py                # Entry point of the application
│   ├── browser_agent          # Module for the browser agent
│   │   ├── __init__.py
│   │   └── agent.py           # Implementation of the browser agent
│   ├── ui                     # Module for the user interface
│   │   ├── __init__.py
│   │   ├── chat_interface.py  # Streamlit chat interface 
│   │   └── model_sidebar.py   # Model selection sidebar
│   └── utils                  # Utility functions
│       ├── __init__.py
│       ├── helpers.py         # Helper utilities
│       └── model_config.py    # Language model configuration
├── pyproject.toml             # Project configuration
├── requirements.txt           # List of dependencies
├── .env                       # Environment variables (create this)
└── README.md                  # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
