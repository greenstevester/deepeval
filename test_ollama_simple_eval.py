"""
Simplified evaluation using Ollama with basic metrics
"""

import requests
import json
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM


class OllamaLLM(DeepEvalBaseLLM):
    """Simple Ollama LLM wrapper"""
    
    def __init__(self, model_name="qwen2.5-coder:3b", base_url="http://10.0.0.125:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt, schema=None):
        """Generate response using Ollama"""
        url = f"{self.base_url}/api/chat"
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        return response.json()["message"]["content"]
    
    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)
    
    def get_model_name(self):
        return self.model_name


# Create Ollama LLM instance
print("Initializing Ollama LLM...")
ollama_llm = OllamaLLM(model_name="qwen2.5-coder:3b")

# Test connection
print("Testing connection...")
try:
    response = ollama_llm.generate("Reply with 'OK' if you receive this.")
    print(f"✓ Connection successful: {response}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Create test cases
print("\nCreating test cases...")
test_cases = [
    LLMTestCase(
        input="What is Python?",
        actual_output="Python is a high-level, interpreted programming language known for simplicity.",
        retrieval_context=[
            "Python is an interpreted, high-level programming language.",
            "Python emphasizes code readability."
        ]
    ),
    LLMTestCase(
        input="What is machine learning?",
        actual_output="Machine learning is a type of AI that enables computers to learn from data.",
        retrieval_context=[
            "Machine learning is a subset of artificial intelligence.",
            "It focuses on building systems that learn from data."
        ]
    )
]

# Create metric
print("Setting up evaluation metric...")
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.5,
    model=ollama_llm,
    include_reason=True
)

# Run evaluation
print("\nRunning evaluation...")
print("="*60)

try:
    results = evaluate(test_cases, [answer_relevancy])
    
    print("\nEvaluation Results:")
    print("-"*60)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Input: {test_case.input}")
        print(f"Output: {test_case.actual_output}")
        
        # The metric should have been measured during evaluate
        if hasattr(answer_relevancy, 'score'):
            print(f"Score: {answer_relevancy.score:.2f}")
            print(f"Success: {'PASS' if answer_relevancy.success else 'FAIL'}")
            if hasattr(answer_relevancy, 'reason'):
                print(f"Reason: {answer_relevancy.reason}")
        
except Exception as e:
    print(f"\nError during evaluation: {e}")
    print("\nTrying individual measurement...")
    
    # Try measuring individually
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case.input}")
        try:
            answer_relevancy.measure(test_case)
            print(f"Score: {answer_relevancy.score:.2f}")
            print(f"Success: {'PASS' if answer_relevancy.success else 'FAIL'}")
        except Exception as e:
            print(f"Error: {e}")

print("\n" + "="*60)
print("Evaluation complete!")
print("\nNote: If you encounter errors, it might be due to:")
print("1. The model's response format")
print("2. The complexity of the metric")
print("3. Try using simpler models like 'mistral:7b' which might handle prompts better")