import Agent

class NormalAgent(Agent.Agent):
    def __init__(self):
        super().__init__()

    def normalagent(self, question, continuous, history):
        response = self.generate_response(question, None, continuous, history)
        print(f"Question: {question}")
        print(f"Response: {response}\n")
        return response
