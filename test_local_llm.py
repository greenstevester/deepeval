"""
End-to-end evaluation example using a locally hosted LLM
This example shows how to use DeepEval with a local LLM server
that exposes an OpenAI-compatible API endpoint.
"""

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, Union
import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    GEval, 
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset, Golden


class LocalLLM(DeepEvalBaseLLM):
    """Custom LLM class for connecting to a local model server"""
    
    def __init__(
        self, 
        model_name: str = "llama3.2",
        base_url: str = "http://10.0.0.125:11434/v1",
        api_key: str = "not-needed"
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        # Don't call parent __init__ to avoid automatic model loading
        
    def load_model(self):
        """Return model identifier"""
        return self.model_name
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """Generate response from local LLM"""
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if schema:
                # Try structured output if the local model supports it
                completion = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                )
                return completion.choices[0].message.parsed
            else:
                # Standard text generation
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,  # Lower temperature for more consistent evaluations
                    max_tokens=1000
                )
                return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling local LLM: {e}")
            # Fallback to standard completion if structured output fails
            if schema:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                return completion.choices[0].message.content
            raise
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """Async generate response from local LLM"""
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if schema:
                completion = await client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                )
                return completion.choices[0].message.parsed
            else:
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling local LLM (async): {e}")
            if schema:
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                return completion.choices[0].message.content
            raise
    
    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model_name


# Initialize the local LLM
local_llm = LocalLLM(
    model_name="llama3.2",  # Adjust based on your local model
    base_url="http://10.0.0.125:11434/v1",
    api_key="not-needed"  # Many local servers don't require API keys
)


# Example 1: Simple test case with GEval metric
def test_simple_evaluation():
    """Test a simple customer service response"""
    
    # Create a custom evaluation metric
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' correctly answers the customer's question based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
        model=local_llm
    )
    
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    
    assert_test(test_case, [correctness_metric])


# Example 2: RAG evaluation with multiple metrics
def test_rag_evaluation():
    """Test a RAG system with multiple evaluation metrics"""
    
    # Define test cases
    test_cases = [
        LLMTestCase(
            input="What are the main benefits of solar energy?",
            actual_output="Solar energy provides clean, renewable power that reduces electricity bills and carbon emissions. It requires minimal maintenance and can increase property values.",
            retrieval_context=[
                "Solar energy is a renewable energy source that comes from the sun.",
                "Benefits include: reduced electricity costs, low maintenance requirements, increased home value, and environmental friendliness.",
                "Solar panels typically last 25-30 years with minimal maintenance needed."
            ]
        ),
        LLMTestCase(
            input="How do I install Python on Windows?",
            actual_output="To install Python on Windows, download the installer from python.org, run it, and make sure to check 'Add Python to PATH' during installation.",
            retrieval_context=[
                "Python installation on Windows: 1. Visit python.org 2. Download the Windows installer 3. Run the installer 4. Check 'Add Python to PATH' 5. Click Install Now",
                "After installation, verify by opening Command Prompt and typing 'python --version'"
            ]
        )
    ]
    
    # Create metrics with local LLM
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=local_llm),
        FaithfulnessMetric(threshold=0.7, model=local_llm),
        ContextualRelevancyMetric(threshold=0.7, model=local_llm)
    ]
    
    # Run evaluation
    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluating test case {i+1}...")
        for metric in metrics:
            metric.measure(test_case)
            print(f"{metric.__class__.__name__}: {metric.score:.2f}")
            if hasattr(metric, 'reason'):
                print(f"Reason: {metric.reason}\n")


# Example 3: Dataset evaluation
def test_dataset_evaluation():
    """Evaluate multiple test cases from a dataset"""
    
    # Create a dataset
    dataset = EvaluationDataset(
        goldens=[
            Golden(
                input="What is machine learning?",
                expected_output="Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
                retrieval_context=["Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."]
            ),
            Golden(
                input="What are the types of machine learning?",
                expected_output="The main types are supervised learning, unsupervised learning, and reinforcement learning.",
                retrieval_context=["Machine learning can be categorized into supervised, unsupervised, and reinforcement learning."]
            )
        ]
    )
    
    # Simulate your LLM application responses
    def mock_llm_app(input_text: str) -> str:
        # In real scenario, this would call your actual LLM application
        responses = {
            "What is machine learning?": "Machine learning is a type of AI where computers learn from data to make predictions without explicit programming.",
            "What are the types of machine learning?": "There are three main types: supervised learning (with labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning through rewards)."
        }
        return responses.get(input_text, "I don't know.")
    
    # Add actual outputs to test cases
    for golden in dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=mock_llm_app(golden.input),
            expected_output=golden.expected_output,
            retrieval_context=golden.retrieval_context
        )
        dataset.add_test_case(test_case)
    
    # Create evaluation metrics
    metrics = [
        GEval(
            name="Correctness",
            criteria="Check if the actual output correctly explains the concept asked in the input.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.7,
            model=local_llm
        ),
        AnswerRelevancyMetric(threshold=0.7, model=local_llm)
    ]
    
    # Evaluate the dataset
    results = evaluate(dataset, metrics)
    return results


# Example 4: Custom criteria evaluation
@pytest.mark.parametrize(
    "test_case",
    [
        LLMTestCase(
            input="How do I reset my password?",
            actual_output="To reset your password, click on 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox.",
            retrieval_context=["Password reset process: 1. Click 'Forgot Password' 2. Enter email 3. Check email 4. Follow reset link"]
        ),
        LLMTestCase(
            input="What payment methods do you accept?",
            actual_output="We accept all major credit cards (Visa, MasterCard, Amex), PayPal, and bank transfers.",
            retrieval_context=["Accepted payment methods: Credit cards (Visa, MasterCard, American Express), PayPal, Wire transfers"]
        )
    ]
)
def test_custom_criteria(test_case):
    """Test with custom evaluation criteria"""
    
    # Create custom metrics
    helpfulness_metric = GEval(
        name="Helpfulness",
        criteria="Evaluate if the response is helpful, clear, and actionable for the user.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=local_llm
    )
    
    completeness_metric = GEval(
        name="Completeness",
        criteria="Check if the response fully addresses all aspects of the user's question.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold=0.7,
        model=local_llm
    )
    
    assert_test(test_case, [helpfulness_metric, completeness_metric])


# Run evaluations without pytest
if __name__ == "__main__":
    print("=" * 50)
    print("Running Local LLM Evaluation Examples")
    print("=" * 50)
    
    print("\n1. Testing connection to local LLM...")
    try:
        # Test the connection
        response = local_llm.generate("Say 'Hello, I'm working!' if you can read this.")
        print(f"✓ Local LLM is responsive: {response}")
    except Exception as e:
        print(f"✗ Error connecting to local LLM: {e}")
        print("Make sure your local LLM server is running at http://10.0.0.125:11434/v1")
        exit(1)
    
    print("\n2. Running RAG evaluation...")
    test_rag_evaluation()
    
    print("\n3. Running dataset evaluation...")
    results = test_dataset_evaluation()
    
    print("\n" + "=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)