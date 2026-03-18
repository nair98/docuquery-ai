class AIEngine:
    def generate_text(self, prompt):
        """
        This is the AI interface.
        Main app calls this function to get AI responses.
        """
        raise NotImplementedError

class CloudAI(AIEngine):
    def generate_text(self, prompt):
        # Here you call the cloud API
        # Example placeholder:
        return f"Cloud response to: {prompt}"
    
class OllamaAI(AIEngine):
    def generate_text(self, prompt):
        # Later, when Ollama is installed, replace this with the actual API call
        return f"Ollama response to: {prompt}"