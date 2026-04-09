# bxAgent
## Requirements
1. It is recommended to use `uv` for project management.
2. You need a LLM. It is developed with the OpenAI interface, but with adjusting the model it should work as well.

## Installation
1. Clone the repository: `git clone git@github.com:BequemeDecke/bxAgent.git`
2. Init the submodules (bxAgent-skills): `git submodule init`
3. Install the dependencies: `uv sync` or `pip install -r requirements.txt`
4. `uv run src/main.py <workspace> <prompt>` or `python src/main.py <workspace> <prompt>`