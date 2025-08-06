"""
Simple test to verify local LLM connection and basic evaluation
"""

from openai import OpenAI
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM


class LocalLLM(DeepEvalBaseLLM):
    """Minimal custom LLM class for local model"""
    
    def __init__(self, model_name="llama3.2", base_url="http://10.0.0.125:11434/v1"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str, schema=None) -> str:
        client = OpenAI(base_url=self.base_url, api_key="dummy")
        
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return completion.choices[0].message.content
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        # For simplicity, using sync version
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        return self.model_name


# Test connection
print("Testing connection to local LLM...")
local_llm = LocalLLM()

try:
    response = local_llm.generate("Hello! Please respond with 'I am working' if you receive this.")
    print(f"✓ Connection successful! Response: {response}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Run a simple evaluation
print("\nRunning evaluation...")

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    retrieval_context=["Paris is the capital and largest city of France."]
)

metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=local_llm,
    include_reason=True
)

# Run evaluation
results = evaluate([test_case], [metric])

print("\nEvaluation Results:")
print(f"Score: {metric.score}")
print(f"Success: {metric.success}")
if hasattr(metric, 'reason'):
    print(f"Reason: {metric.reason}")