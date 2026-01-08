from configs.config import Config
import openai
import ollama
import os


class LLMChat:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm.provider
        self.model = config.llm.model
        self.client = (
            openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if config.llm.provider == "openai"
            else None
        )

    def chat(self, messages):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content
        elif self.provider == "ollama":
            response = ollama.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        else:
            raise ValueError(f"Invalid provider: {self.provider}")
