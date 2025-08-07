"""
Working local LLM integration with custom metrics that handle text responses
"""

import requests
import deepeval
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import BaseMetric
import re


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
    """Custom relevancy metric that works with text responses"""
    
    def __init__(self, model: DeepEvalBaseLLM, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.include_reason = True
        
    @property
    def __name__(self):
        return "Simple Relevancy"
    
    def measure(self, test_case: LLMTestCase):
        prompt = f"""Rate how relevant the answer is to the question on a scale of 0-10.

Question: {test_case.input}
Answer: {test_case.actual_output}

Consider:
- Does the answer address the question?
- Is the information helpful?
- Is it appropriate for the context?

Respond with:
SCORE: [number 0-10]
REASON: [brief explanation]"""
        
        try:
            response = self.model.generate(prompt)
            
            # Parse score
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1)) / 10.0
            else:
                score = 0.5
            
            # Parse reason
            reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = "Could not parse evaluation reason"
            
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


class SimpleCorrectnessMetric(BaseMetric):
    """Custom correctness metric"""
    
    def __init__(self, model: DeepEvalBaseLLM, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.include_reason = True
        
    @property
    def __name__(self):
        return "Simple Correctness"
    
    def measure(self, test_case: LLMTestCase):
        prompt = f"""Compare the actual answer with the expected answer and rate correctness 0-10.

Question: {test_case.input}
Actual Answer: {test_case.actual_output}
Expected Answer: {test_case.expected_output or "N/A"}

Rate how correct the actual answer is:
- 0: Completely wrong
- 5: Partially correct
- 10: Perfectly correct

SCORE: [number 0-10]
REASON: [explanation]"""
        
        try:
            response = self.model.generate(prompt)
            
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5
            
            reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Could not parse reason"
            
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


# Initialize LLM
local_llm = OllamaLLM()


# Your actual application
def customer_support_app(question: str) -> str:
    """Your actual LLM application"""
    
    knowledge = {
        "refund": "We offer a 30-day full refund policy for all items. Items must be unworn.",
        "shipping": "Standard shipping takes 5-7 business days. Express shipping available.",
        "contact": "Contact us at support@example.com or 1-800-HELP for assistance."
    }
    
    # Simple context selection
    context = knowledge["contact"]  # default
    if any(word in question.lower() for word in ["refund", "return", "fit", "size"]):
        context = knowledge["refund"]
    elif any(word in question.lower() for word in ["ship", "delivery", "arrive"]):
        context = knowledge["shipping"]
    
    prompt = f"""You are a helpful customer service assistant.

Policy: {context}

Customer Question: {question}

Provide a helpful response based on the policy:"""
    
    return local_llm.generate(prompt)


def test_refund_question():
    """Test refund policy question"""
    
    question = "What if these shoes don't fit?"
    actual_output = customer_support_app(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="We offer a 30-day full refund policy for all items.",
    )
    
    # Use custom metrics that work with local LLM
    relevancy_metric = SimpleRelevancyMetric(model=local_llm, threshold=0.6)
    correctness_metric = SimpleCorrectnessMetric(model=local_llm, threshold=0.6)
    
    assert_test(test_case, [relevancy_metric, correctness_metric])


def test_shipping_question():
    """Test shipping question"""
    
    question = "How long does shipping take?"
    actual_output = customer_support_app(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="Standard shipping takes 5-7 business days.",
    )
    
    relevancy_metric = SimpleRelevancyMetric(model=local_llm, threshold=0.6)
    
    assert_test(test_case, [relevancy_metric])


def test_contact_question():
    """Test contact information question"""
    
    question = "How can I contact customer service?"
    actual_output = customer_support_app(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
    )
    
    relevancy_metric = SimpleRelevancyMetric(model=local_llm, threshold=0.6)
    
    assert_test(test_case, [relevancy_metric])


@deepeval.log_hyperparameters
def hyperparameters():
    return {
        "model": "qwen2.5-coder:3b",
        "temperature": 0.1,
        "application": "customer_support_v2",
        "metrics": "custom_text_based"
    }


if __name__ == "__main__":
    print("Working Customer Support App with Local LLM")
    print("=" * 60)
    
    # Test the application
    test_questions = [
        "What if these shoes don't fit?",
        "How long does shipping take?",
        "How can I contact you?",
        "What's your return policy?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = customer_support_app(question)
        print(f"A: {answer}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("To run DeepEval tests:")
    print("deepeval test run test_working_local_llm.py")
    
    # Test individual functions
    print("\nTesting individual functions...")
    try:
        test_refund_question()
        print("✓ Refund test passed")
    except Exception as e:
        print(f"✗ Refund test failed: {e}")
    
    try:
        test_shipping_question()
        print("✓ Shipping test passed")
    except Exception as e:
        print(f"✗ Shipping test failed: {e}")
    
    try:
        test_contact_question()
        print("✓ Contact test passed")
    except Exception as e:
        print(f"✗ Contact test failed: {e}")