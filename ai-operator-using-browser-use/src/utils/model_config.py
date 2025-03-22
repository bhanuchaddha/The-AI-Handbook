MODEL_PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini"],
        "api_key_env": "OPENAI_API_KEY"
    },
    "Claude": {
        "models": ["claude-3-7-sonnet"],
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "DeepSeek": {
        "models": ["deepseek-v3", "deepseek-r1"],
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    "Ollama": {
        "models": ["deepseek-r1"],
        "api_key_env": None,
        "warning": "Please make sure the selected model is running locally with Ollama."
    }
}
