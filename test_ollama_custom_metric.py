"""
Custom metrics for Ollama evaluation that handle text responses better
"""

import requests
import json
from typing import List, Optional
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import BaseMetric
# from deepeval.scorer import Scorer  # Not needed for this example


class OllamaLLM(DeepEvalBaseLLM):
    """Ollama LLM wrapper"""
    
    def __init__(self, model_name="qwen2.5-coder:3b", base_url="http://10.0.0.125:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt, schema=None):
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


class SimpleRelevancyMetric(BaseMetric):
    """Custom relevancy metric that handles text responses"""
    
    def __init__(self, model: DeepEvalBaseLLM, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.include_reason = True
        
    @property
    def __name__(self):
        return "Simple Relevancy"
    
    def measure(self, test_case: LLMTestCase):
        """Measure relevancy using a simple prompt"""
        
        # Create evaluation prompt
        prompt = f"""
You are an expert evaluator. Please evaluate if the given output is relevant to the input question.

Input Question: {test_case.input}
Output Answer: {test_case.actual_output}

Rate the relevancy on a scale of 0 to 10, where:
- 0 means completely irrelevant
- 5 means somewhat relevant
- 10 means perfectly relevant

Provide your response in this format:
SCORE: [number]
REASON: [your explanation]
"""
        
        try:
            # Get evaluation from model
            response = self.model.generate(prompt)
            
            # Parse response
            lines = response.strip().split('\n')
            score_line = None
            reason_line = None
            
            for line in lines:
                if line.upper().startswith('SCORE:'):
                    score_line = line
                elif line.upper().startswith('REASON:'):
                    reason_line = line
            
            # Extract score
            if score_line:
                score_str = score_line.split(':', 1)[1].strip()
                # Extract number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', score_str)
                if numbers:
                    score = float(numbers[0]) / 10.0  # Normalize to 0-1
                else:
                    score = 0.5  # Default score
            else:
                score = 0.5
            
            # Extract reason
            if reason_line:
                reason = reason_line.split(':', 1)[1].strip()
            else:
                reason = "Could not parse evaluation reason"
            
            self.score = score
            self.reason = reason
            self.success = score >= self.threshold
            
            return self.score
            
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error during evaluation: {str(e)}"
            self.success = False
            return self.score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        return self.success


class SimpleFaithfulnessMetric(BaseMetric):
    """Custom faithfulness metric for context-based evaluation"""
    
    def __init__(self, model: DeepEvalBaseLLM, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.include_reason = True
        
    @property
    def __name__(self):
        return "Simple Faithfulness"
    
    def measure(self, test_case: LLMTestCase):
        """Measure if output is faithful to the context"""
        
        if not test_case.retrieval_context:
            self.score = 1.0
            self.reason = "No context provided to check against"
            self.success = True
            return self.score
        
        context_str = "\n".join(test_case.retrieval_context)
        
        prompt = f"""
You are an expert evaluator. Check if the output is faithful to the given context.

Context:
{context_str}

Output: {test_case.actual_output}

Is the output faithful to the context? Does it contain any information not present in the context?

Rate faithfulness on a scale of 0 to 10:
- 0 means completely unfaithful (contradicts context or adds false information)
- 5 means partially faithful
- 10 means completely faithful (only uses information from context)

Respond in this format:
SCORE: [number]
REASON: [explanation]
"""
        
        try:
            response = self.model.generate(prompt)
            
            # Parse response
            lines = response.strip().split('\n')
            score = 0.5  # default
            reason = "Could not parse response"
            
            for line in lines:
                if line.upper().startswith('SCORE:'):
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        score = float(numbers[0]) / 10.0
                elif line.upper().startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()
            
            self.score = score
            self.reason = reason
            self.success = score >= self.threshold
            
            return self.score
            
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error: {str(e)}"
            self.success = False
            return self.score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        return self.success


# Initialize Ollama
print("Setting up Ollama LLM...")
ollama_llm = OllamaLLM(model_name="qwen2.5-coder:3b")

# Test connection
print("Testing connection...")
response = ollama_llm.generate("Say 'Connected!' if you receive this.")
print(f"✓ Response: {response}\n")

# Create test cases
test_cases = [
    LLMTestCase(
        input="What is Python?",
        actual_output="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        retrieval_context=[
            "Python is an interpreted, high-level programming language.",
            "Python emphasizes code readability and simplicity."
        ]
    ),
    LLMTestCase(
        input="What are Python lists?",
        actual_output="Python lists are ordered, mutable collections that can store multiple items of different types.",
        retrieval_context=[
            "Lists in Python are ordered collections of items.",
            "Lists are mutable, meaning they can be modified after creation.",
            "Lists can contain elements of different data types."
        ]
    ),
    LLMTestCase(
        input="How do I install Python?",
        actual_output="To install Python, download it from python.org and run the installer.",
        retrieval_context=[
            "Python can be downloaded from the official website python.org",
            "Installation involves running the downloaded installer",
            "Make sure to add Python to PATH during installation"
        ]
    )
]

# Create custom metrics
relevancy_metric = SimpleRelevancyMetric(model=ollama_llm, threshold=0.6)
faithfulness_metric = SimpleFaithfulnessMetric(model=ollama_llm, threshold=0.6)

# Evaluate each test case
print("Running Evaluation")
print("=" * 60)

for i, test_case in enumerate(test_cases):
    print(f"\nTest Case {i+1}: {test_case.input}")
    print("-" * 40)
    
    # Measure relevancy
    print("Evaluating relevancy...")
    relevancy_metric.measure(test_case)
    print(f"Relevancy Score: {relevancy_metric.score:.2f} ({'PASS' if relevancy_metric.success else 'FAIL'})")
    print(f"Reason: {relevancy_metric.reason}")
    
    # Measure faithfulness
    print("\nEvaluating faithfulness...")
    faithfulness_metric.measure(test_case)
    print(f"Faithfulness Score: {faithfulness_metric.score:.2f} ({'PASS' if faithfulness_metric.success else 'FAIL'})")
    print(f"Reason: {faithfulness_metric.reason}")

print("\n" + "=" * 60)
print("Evaluation Complete!")

# You can also use evaluate() function
print("\nRunning batch evaluation with evaluate()...")
metrics = [relevancy_metric, faithfulness_metric]
results = evaluate(test_cases[:1], metrics)

print("\nTo use with pytest:")
print("1. Save this as a test file")
print("2. Run: deepeval test run <filename>")
print("\nAvailable models on your server:")
print("- qwen2.5-coder:3b (current)")
print("- mistral:7b")
print("- qwen:7b")