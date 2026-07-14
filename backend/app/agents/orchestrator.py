from agents.router import RouterAgent

from agents.operation_agent import OperationAgent

from agents.troubleshooting_agent import TroubleshootingAgent

from agents.maintenance_agent import MaintenanceAgent


class AgentOrchestrator:

    def __init__(self):

        self.router = RouterAgent()

        self.operation = OperationAgent()

        self.troubleshooting = TroubleshootingAgent()

        self.maintenance = MaintenanceAgent()

    def process(self, question, context):

        intent = self.router.detect(question)

        if intent == "operation":
            return self.operation.run(question, context)

        elif intent == "troubleshooting":
            return self.troubleshooting.run(question, context)

        else:
            return self.maintenance.run(question, context)