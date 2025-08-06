"""
End-to-end evaluation using Ollama locally hosted LLM
"""

import requests
import json
from typing import Optional, Union
from pydantic import BaseModel
from deepeval import evaluate, assert_test
from deepeval.metrics import (
    GEval, 
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset, Golden
import pytest


class OllamaLLM(DeepEvalBaseLLM):
    """Custom LLM class for Ollama"""
    
    def __init__(
        self, 
        model_name: str = "qwen2.5-coder:3b",  # Using one of your available models
        base_url: str = "http://10.0.0.125:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Generate response using Ollama API"""
        
        # Ollama chat API endpoint
        url = f"{self.base_url}/api/chat"
        
        # Prepare the request
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for consistent evaluation
                "num_predict": 1000
            }
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format: {response.text}")
            raise
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Async version - using sync for simplicity"""
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        return self.model_name


# Initialize Ollama LLM with your available model
ollama_llm = OllamaLLM(
    model_name="qwen2.5-coder:3b",  # You can change to any available model
    base_url="http://10.0.0.125:11434"
)


# Test 1: Simple connection test
def test_ollama_connection():
    """Test if Ollama is responding"""
    print("Testing Ollama connection...")
    try:
        response = ollama_llm.generate("Say 'Hello, I am working!' if you can read this.")
        print(f"✓ Ollama responded: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


# Test 2: Simple evaluation
def test_simple_evaluation():
    """Run a simple evaluation test"""
    
    test_case = LLMTestCase(
        input="What is 2 + 2?",
        actual_output="2 + 2 equals 4.",
        expected_output="The answer is 4."
    )
    
    # Create a simple correctness metric
    correctness_metric = GEval(
        name="Mathematical Correctness",
        criteria="Check if the actual output gives the correct mathematical answer.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=ollama_llm
    )
    
    # Measure
    correctness_metric.measure(test_case)
    print(f"\nCorrectness Score: {correctness_metric.score}")
    print(f"Success: {correctness_metric.success}")
    if hasattr(correctness_metric, 'reason'):
        print(f"Reason: {correctness_metric.reason}")


# Test 3: RAG evaluation
def test_rag_evaluation():
    """Test RAG metrics with context"""
    
    test_cases = [
        LLMTestCase(
            input="What is Python?",
            actual_output="Python is a high-level, interpreted programming language known for its simplicity and readability.",
            retrieval_context=[
                "Python is an interpreted, high-level programming language.",
                "Python emphasizes code readability and simplicity.",
                "It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
            ]
        ),
        LLMTestCase(
            input="How do I create a list in Python?",
            actual_output="To create a list in Python, use square brackets like this: my_list = [1, 2, 3] or my_list = ['a', 'b', 'c']",
            retrieval_context=[
                "Lists in Python are created using square brackets []",
                "Example: my_list = [1, 2, 3, 4, 5]",
                "Lists can contain mixed data types: mixed_list = [1, 'hello', 3.14]"
            ]
        )
    ]
    
    # Create metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=ollama_llm),
        FaithfulnessMetric(threshold=0.5, model=ollama_llm),
        ContextualRelevancyMetric(threshold=0.5, model=ollama_llm)
    ]
    
    print("\n" + "="*50)
    print("RAG Evaluation Results")
    print("="*50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case.input}")
        print("-" * 30)
        
        for metric in metrics:
            try:
                metric.measure(test_case)
                print(f"{metric.__class__.__name__}: {metric.score:.2f} ({'PASS' if metric.success else 'FAIL'})")
                if hasattr(metric, 'reason') and metric.reason:
                    print(f"  Reason: {metric.reason[:100]}...")
            except Exception as e:
                print(f"{metric.__class__.__name__}: Error - {str(e)[:100]}...")


# Test 4: Dataset evaluation with pytest
@pytest.mark.parametrize(
    "test_case",
    [
        LLMTestCase(
            input="What is machine learning?",
            actual_output="Machine learning is a type of artificial intelligence that enables computers to learn from data without explicit programming.",
            expected_output="Machine learning is a subset of AI that allows systems to learn from data.",
            retrieval_context=["Machine learning is a branch of AI focused on building systems that learn from data."]
        ),
        LLMTestCase(
            input="What is a neural network?",
            actual_output="A neural network is a computing system inspired by biological neural networks that uses interconnected nodes to process information.",
            expected_output="A neural network is a series of algorithms that attempts to recognize patterns in data.",
            retrieval_context=["Neural networks are computing systems inspired by the biological neural networks in animal brains."]
        )
    ]
)
def test_pytest_evaluation(test_case):
    """Test case for pytest integration"""
    
    # Use G-Eval for flexible evaluation
    understanding_metric = GEval(
        name="Conceptual Understanding",
        criteria="Evaluate if the actual output demonstrates a correct understanding of the concept asked about in the input.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.6,
        model=ollama_llm
    )
    
    assert_test(test_case, [understanding_metric])


# Main execution
if __name__ == "__main__":
    print("DeepEval with Ollama - Local LLM Evaluation")
    print("=" * 60)
    
    # Test connection first
    if not test_ollama_connection():
        print("\nPlease ensure Ollama is running at http://10.0.0.125:11434")
        exit(1)
    
    # Run simple evaluation
    print("\n" + "="*60)
    print("Running Simple Evaluation")
    print("="*60)
    test_simple_evaluation()
    
    # Run RAG evaluation
    test_rag_evaluation()
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    print("\nTo run pytest integration tests, use:")
    print("  deepeval test run test_ollama_eval.py::test_pytest_evaluation -v")
    
    print("\nAvailable models on your server:")
    print("  - qwen2.5-coder:3b")
    print("  - qwen2.5-coder:1.5b") 
    print("  - mistral:7b")
    print("  - qwen:7b")