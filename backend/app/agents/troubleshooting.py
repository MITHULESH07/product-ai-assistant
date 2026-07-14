from llm.ollama_client import OllamaClient


class TroubleshootingAgent:

    def __init__(self):
        self.llm = OllamaClient()

    def run(self, question, context):

        return self.llm.ask(question, context)