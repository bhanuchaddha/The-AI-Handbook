# Agentic Chat with Browser Using BrowserUse

This project is a Python application that utilizes Streamlit as a user interface to send chat instructions to a browser agent using the Browser Use library. It allows users to interact with a browser agent and receive responses in real-time.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/bhanuchaddha/agentic-chat-with-browser.git
   cd agentic-chat-with-browser
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
streamlit run src/main.py
```

This will start the Streamlit server and open the application in your default web browser. You can then interact with the chat interface to send instructions to the browser agent.

## Project Structure

```
agentic-chat-with-Browser-using-BrowserUse
├── src
│   ├── main.py                # Entry point of the application
│   ├── browser_agent          # Module for the browser agent
│   │   ├── __init__.py
│   │   └── agent.py           # Implementation of the browser agent
│   ├── ui                     # Module for the user interface
│   │   ├── __init__.py
│   │   └── chat_interface.py   # Streamlit chat interface
│   └── utils                  # Utility functions
│       └── helpers.py
├── pyproject.toml             # Project configuration
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
