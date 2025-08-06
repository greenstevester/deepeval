"""
Modified test_example.py to work with local Ollama LLM
This example shows how to integrate your actual LLM application with DeepEval
"""

import pytest
import deepeval
import requests
from typing import List, Optional
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM

# To run this file: deepeval test run test_example_local_llm.py


class OllamaLLM(DeepEvalBaseLLM):
    """Custom LLM wrapper for Ollama"""
    
    def __init__(self, model_name="qwen2.5-coder:3b", base_url="http://10.0.0.125:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str, schema=None) -> str:
        """Generate response using Ollama API"""
        url = f"{self.base_url}/api/chat"
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            raise
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        return self.model_name


# Initialize your local LLM
local_llm = OllamaLLM(model_name="qwen2.5-coder:3b")


# Your actual LLM application - Customer Support Chatbot
class CustomerSupportBot:
    """
    Your actual LLM application that uses the local Ollama model
    This is a simple RAG-based customer support chatbot
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        # Knowledge base for the chatbot
        self.knowledge_base = {
            "refund_policy": "We offer a 30-day full refund policy. No questions asked. Customers can return items within 30 days of purchase for a complete refund.",
            "shipping_info": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available for an additional $10.",
            "size_guide": "Our shoes run true to size. If between sizes, we recommend ordering the larger size. Size exchanges are free.",
            "customer_service": "Our customer service is available 24/7 via chat, email at support@example.com, or phone at 1-800-SHOES.",
            "warranty": "All shoes come with a 1-year warranty against manufacturing defects."
        }
    
    def get_relevant_context(self, query: str) -> List[str]:
        """Simple keyword-based context retrieval"""
        query_lower = query.lower()
        relevant_contexts = []
        
        if any(word in query_lower for word in ["refund", "return", "money back", "fit"]):
            relevant_contexts.append(self.knowledge_base["refund_policy"])
        if any(word in query_lower for word in ["ship", "delivery", "arrive"]):
            relevant_contexts.append(self.knowledge_base["shipping_info"])
        if any(word in query_lower for word in ["size", "fit", "big", "small"]):
            relevant_contexts.append(self.knowledge_base["size_guide"])
        if any(word in query_lower for word in ["contact", "help", "support"]):
            relevant_contexts.append(self.knowledge_base["customer_service"])
        if any(word in query_lower for word in ["warranty", "defect", "quality"]):
            relevant_contexts.append(self.knowledge_base["warranty"])
        
        return relevant_contexts if relevant_contexts else [self.knowledge_base["customer_service"]]
    
    def answer_question(self, question: str) -> tuple[str, List[str]]:
        """
        Main method to answer customer questions
        Returns: (answer, contexts_used)
        """
        # Get relevant context
        contexts = self.get_relevant_context(question)
        context_str = "\n".join(contexts)
        
        # Create prompt for the LLM
        prompt = f"""You are a helpful customer support assistant for an online shoe store.
        
Use the following information to answer the customer's question:

Context:
{context_str}

Customer Question: {question}

Please provide a helpful, friendly, and accurate response based on the context provided. 
If the information isn't in the context, politely say you'll need to check with the team."""
        
        # Get response from LLM
        answer = self.llm.generate(prompt)
        
        return answer, contexts


# Initialize your chatbot
chatbot = CustomerSupportBot(local_llm)


# Create test dataset with real customer queries
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What if these shoes don't fit?",
            expected_output="You're eligible for a free full refund within 30 days of purchase. We also offer free size exchanges.",
            retrieval_context=["We offer a 30-day full refund policy. No questions asked. Customers can return items within 30 days of purchase for a complete refund."]
        ),
        Golden(
            input="How long does shipping take?",
            expected_output="Standard shipping takes 5-7 business days. Express shipping is available for 2-3 days delivery.",
            retrieval_context=["Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available for an additional $10."]
        ),
        Golden(
            input="Do you have a warranty?",
            expected_output="Yes, all shoes come with a 1-year warranty against manufacturing defects.",
            retrieval_context=["All shoes come with a 1-year warranty against manufacturing defects."]
        ),
        Golden(
            input="I need help with my order",
            expected_output="Our customer service team is available 24/7 to help you via chat, email, or phone.",
            retrieval_context=["Our customer service is available 24/7 via chat, email at support@example.com, or phone at 1-800-SHOES."]
        )
    ]
)


# Add actual outputs from your LLM application to the dataset
for golden in dataset.goldens:
    # Get actual response from your chatbot
    actual_output, contexts_used = chatbot.answer_question(golden.input)
    
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=contexts_used  # Use the actual contexts from your app
    )
    dataset.add_test_case(test_case)


@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_customer_support_chatbot(test_case: LLMTestCase):
    """Test the customer support chatbot responses"""
    
    # Use metrics with your local LLM
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=local_llm,
        include_reason=True
    )
    
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the actual output correctly answers the customer's question and provides accurate information based on the expected output.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=local_llm
    )
    
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=local_llm,
        include_reason=True
    )
    
    # Run assertions
    assert_test(test_case, [answer_relevancy_metric, correctness_metric, faithfulness_metric])


# Individual test for specific scenarios
def test_refund_policy():
    """Test specific refund policy responses"""
    
    question = "I bought shoes last week but they're too small. Can I return them?"
    actual_output, contexts = chatbot.answer_question(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="Yes, you can return them within 30 days for a full refund or exchange for a different size.",
        retrieval_context=contexts
    )
    
    # Create custom metric for helpfulness
    helpfulness_metric = GEval(
        name="Helpfulness",
        criteria="Evaluate if the response is helpful, clear, and provides actionable information for the customer.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=local_llm
    )
    
    assert_test(test_case, [helpfulness_metric])


# Log hyperparameters for tracking
@deepeval.log_hyperparameters
def hyperparameters():
    return {
        "model": "qwen2.5-coder:3b",
        "prompt_template": "customer_support_v1",
        "temperature": 0.7,
        "max_tokens": 500,
        "knowledge_base_size": 5,
        "retrieval_method": "keyword_based"
    }


if __name__ == "__main__":
    """Run a quick test of the chatbot"""
    print("Testing Customer Support Chatbot with Local LLM")
    print("=" * 60)
    
    # Test the chatbot
    test_questions = [
        "What if these shoes don't fit?",
        "How can I contact customer service?",
        "Do you ship internationally?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer, contexts = chatbot.answer_question(question)
        print(f"A: {answer}")
        print(f"Contexts used: {len(contexts)}")
    
    print("\n" + "=" * 60)
    print("To run full evaluation:")
    print("deepeval test run test_example_local_llm.py")