tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code to analyze data or create visualizations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use plt.savefig() for charts.",
                    },
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra pip requirements to install before executing.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation of what this code does.",
                    },
                },
                "required": ["code"],
            },
        },
    }
]
