from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

from Agent.demo1.tools import list_files, read_file, rename_file


ollama_model = OpenAIModel(
    model_name='llama3.2',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

agent = Agent(
    ollama_model,
    system_prompt='You are an experienced programmer.',
    tools=[
        list_files,
        read_file,
        rename_file
    ]
)


def main() -> None:
    history = []
    while True:
        user_input = input("Input: ")
        if user_input.lower() == "exit":
            break
        result = agent.run_sync(user_input, message_history=history)
        history = list(result.all_messages())
        print(result.output)


if __name__ == "__main__":
    main()