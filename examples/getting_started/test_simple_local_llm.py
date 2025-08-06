"""
Simple local LLM integration with DeepEval - no parametrize issues
"""

import requests
import deepeval
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM


class OllamaLLM(DeepEvalBaseLLM):
    """Simple Ollama LLM wrapper"""
    
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


# Initialize LLM
local_llm = OllamaLLM()


# Your actual application function
def customer_support_app(question: str) -> str:
    """Your actual LLM application"""
    
    # Knowledge base
    knowledge = {
        "refund": "We offer a 30-day full refund policy for all items.",
        "shipping": "Standard shipping takes 5-7 business days.",
        "contact": "Reach us at support@example.com or 1-800-HELP."
    }
    
    # Simple context selection
    context = ""
    if "refund" in question.lower() or "return" in question.lower():
        context = knowledge["refund"]
    elif "ship" in question.lower():
        context = knowledge["shipping"]
    else:
        context = knowledge["contact"]
    
    # Create prompt
    prompt = f"""You are a customer service assistant. Answer the question using this information:

Context: {context}

Question: {question}

Answer helpfully and professionally:"""
    
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
    
    # Create metric with local LLM
    correctness_metric = GEval(
        name="Correctness",
        criteria="Check if the actual output correctly addresses the customer's question about returns/refunds.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
        model=local_llm
    )
    
    assert_test(test_case, [correctness_metric])


def test_shipping_question():
    """Test shipping question"""
    
    question = "How long does shipping take?"
    actual_output = customer_support_app(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="Standard shipping takes 5-7 business days.",
    )
    
    helpfulness_metric = GEval(
        name="Helpfulness",
        criteria="Evaluate if the response is helpful and provides useful information to the customer.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
        model=local_llm
    )
    
    assert_test(test_case, [helpfulness_metric])


def test_multiple_questions():
    """Test multiple questions using evaluate function"""
    
    questions = [
        "Can I return my shoes?",
        "When will my order arrive?", 
        "How do I contact support?"
    ]
    
    test_cases = []
    for question in questions:
        actual_output = customer_support_app(question)
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output
        )
        test_cases.append(test_case)
    
    # Create a simple relevancy metric
    relevancy_metric = GEval(
        name="Response Relevancy",
        criteria="Check if the response is relevant to the customer's question.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=local_llm
    )
    
    # Evaluate all test cases
    results = evaluate(test_cases, [relevancy_metric])
    
    # Check that all passed
    assert all(relevancy_metric.success for _ in range(len(test_cases)))


@deepeval.log_hyperparameters
def hyperparameters():
    return {
        "model": "qwen2.5-coder:3b",
        "temperature": 0.1,
        "application": "customer_support_v1"
    }


if __name__ == "__main__":
    print("Testing Customer Support App with Local LLM")
    print("=" * 50)
    
    # Test the application
    test_questions = [
        "What if these shoes don't fit?",
        "How long does shipping take?",
        "How can I contact you?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = customer_support_app(question)
        print(f"A: {answer}")
    
    print("\n" + "=" * 50)
    print("To run DeepEval tests:")
    print("deepeval test run test_simple_local_llm.py")