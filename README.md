# Revent

This repository contains a small example program that demonstrates how to
combine two OpenAI models to reduce inference cost. A powerful model
first generates a reasoning plan, and then a cheaper model follows the plan
to produce the final answer.

## Requirements
- Python 3.8+
- `openai` Python package
- An OpenAI API key in the environment variable `OPENAI_API_KEY`

Install dependencies:
```
pip install openai
```

## Usage
Run the `plan_reason.py` script with your question or request:
```
python plan_reason.py "Explain the theory of relativity"
```

The script first asks `gpt-4o` for a reasoning plan, then uses
`gpt-3.5-turbo` to generate the answer following that plan.
