"""
Advanced RAG Application with Local LLM and DeepEval Integration
This example shows a more sophisticated RAG pipeline with vector search simulation
"""

import pytest
import deepeval
import requests
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from deepeval import assert_test, evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.models import DeepEvalBaseLLM


@dataclass
class Document:
    """Represents a document in our knowledge base"""
    id: str
    content: str
    metadata: Dict[str, str]
    embedding: List[float] = None


class OllamaLLM(DeepEvalBaseLLM):
    """Ollama LLM wrapper"""
    
    def __init__(self, model_name="qwen2.5-coder:3b", base_url="http://10.0.0.125:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str, schema=None) -> str:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 500}
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            raise
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        return self.model_name


class RAGApplication:
    """
    A complete RAG application with document store, retrieval, and generation
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.documents = self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> List[Document]:
        """Initialize with a comprehensive knowledge base"""
        docs = [
            # Product Information
            Document(
                id="doc1",
                content="Our premium running shoes feature advanced cushioning technology with responsive foam that provides excellent energy return. The breathable mesh upper ensures comfort during long runs.",
                metadata={"category": "product", "type": "running_shoes"}
            ),
            Document(
                id="doc2",
                content="The hiking boots are waterproof with Gore-Tex lining and Vibram outsoles for superior traction on wet and dry surfaces. Ankle support is reinforced for challenging terrain.",
                metadata={"category": "product", "type": "hiking_boots"}
            ),
            
            # Policies
            Document(
                id="doc3",
                content="Return Policy: Items can be returned within 30 days of purchase for a full refund. Items must be unworn and in original packaging. Return shipping is free for all customers.",
                metadata={"category": "policy", "type": "returns"}
            ),
            Document(
                id="doc4",
                content="Shipping Policy: Standard shipping (5-7 business days) is free on orders over $50. Express shipping (2-3 days) costs $10. International shipping available to select countries.",
                metadata={"category": "policy", "type": "shipping"}
            ),
            Document(
                id="doc5",
                content="Warranty: All footwear comes with a 1-year warranty covering manufacturing defects. Normal wear and tear is not covered. Warranty claims must include proof of purchase.",
                metadata={"category": "policy", "type": "warranty"}
            ),
            
            # Care Instructions
            Document(
                id="doc6",
                content="Shoe Care: Clean shoes with mild soap and water. Allow to air dry away from direct heat. Use appropriate waterproofing spray for leather and suede materials.",
                metadata={"category": "care", "type": "maintenance"}
            ),
            
            # Sizing
            Document(
                id="doc7",
                content="Size Guide: Our shoes run true to size for most customers. Wide sizes available in select styles. European sizes: EU 36-47. US sizes: Men's 5-14, Women's 5-12.",
                metadata={"category": "sizing", "type": "guide"}
            )
        ]
        
        # Simulate embeddings (in real app, use actual embedding model)
        for i, doc in enumerate(docs):
            doc.embedding = [0.1 * i] * 384  # Simulated 384-dim embedding
            
        return docs
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve top-k relevant documents
        In production, this would use vector similarity search
        """
        # For demo, use keyword matching with scoring
        query_lower = query.lower()
        scores = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc.content.lower()
            
            # Simple keyword scoring
            keywords = query_lower.split()
            for keyword in keywords:
                if keyword in content_lower:
                    score += content_lower.count(keyword)
            
            # Boost score for matching metadata
            if any(keyword in doc.metadata.get('type', '').lower() for keyword in keywords):
                score += 2
                
            scores.append((score, doc))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k] if _ > 0]
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using retrieved contexts"""
        
        if not contexts:
            return "I couldn't find relevant information to answer your question. Please contact our customer service for assistance."
        
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a helpful customer service assistant for a shoe store. 
Answer the customer's question based ONLY on the provided context. Be accurate and helpful.

Context:
{context_str}

Customer Question: {query}

Instructions:
- Answer based only on the provided context
- Be concise but complete
- If the context doesn't contain the answer, say so politely
- Maintain a friendly, professional tone

Answer:"""
        
        return self.llm.generate(prompt)
    
    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        Main RAG pipeline: retrieve -> generate
        Returns: (answer, contexts_used)
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, k=3)
        
        # Extract contexts
        contexts = [doc.content for doc in retrieved_docs]
        
        # Generate answer
        answer = self.generate_answer(question, contexts)
        
        return answer, contexts


# Initialize components
local_llm = OllamaLLM(model_name="qwen2.5-coder:3b")
rag_app = RAGApplication(local_llm)


# Create comprehensive test dataset
dataset = EvaluationDataset(
    goldens=[
        # Product queries
        Golden(
            input="Tell me about your running shoes",
            expected_output="Our running shoes feature advanced cushioning technology with responsive foam for excellent energy return and a breathable mesh upper for comfort.",
            retrieval_context=["Our premium running shoes feature advanced cushioning technology with responsive foam that provides excellent energy return. The breathable mesh upper ensures comfort during long runs."]
        ),
        Golden(
            input="Are the hiking boots waterproof?",
            expected_output="Yes, our hiking boots are waterproof with Gore-Tex lining and also feature Vibram outsoles for superior traction.",
            retrieval_context=["The hiking boots are waterproof with Gore-Tex lining and Vibram outsoles for superior traction on wet and dry surfaces. Ankle support is reinforced for challenging terrain."]
        ),
        
        # Policy queries
        Golden(
            input="What's your return policy?",
            expected_output="You can return items within 30 days for a full refund. Items must be unworn and in original packaging. Return shipping is free.",
            retrieval_context=["Return Policy: Items can be returned within 30 days of purchase for a full refund. Items must be unworn and in original packaging. Return shipping is free for all customers."]
        ),
        Golden(
            input="How long does shipping take?",
            expected_output="Standard shipping takes 5-7 business days and is free on orders over $50. Express shipping (2-3 days) is available for $10.",
            retrieval_context=["Shipping Policy: Standard shipping (5-7 business days) is free on orders over $50. Express shipping (2-3 days) costs $10. International shipping available to select countries."]
        ),
        
        # Complex queries
        Golden(
            input="I need waterproof shoes with good warranty. What do you recommend?",
            expected_output="I recommend our waterproof hiking boots with Gore-Tex lining. All our footwear comes with a 1-year warranty covering manufacturing defects.",
            retrieval_context=[
                "The hiking boots are waterproof with Gore-Tex lining and Vibram outsoles for superior traction on wet and dry surfaces. Ankle support is reinforced for challenging terrain.",
                "Warranty: All footwear comes with a 1-year warranty covering manufacturing defects. Normal wear and tear is not covered. Warranty claims must include proof of purchase."
            ]
        )
    ]
)


# Generate actual outputs for test cases
for golden in dataset.goldens:
    actual_output, contexts = rag_app.query(golden.input)
    
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=contexts
    )
    dataset.add_test_case(test_case)


@pytest.mark.parametrize("test_case", dataset)
def test_rag_pipeline(test_case: LLMTestCase):
    """Comprehensive RAG pipeline testing"""
    
    # Initialize metrics with local LLM
    metrics = [
        # Response quality metrics
        AnswerRelevancyMetric(threshold=0.7, model=local_llm),
        FaithfulnessMetric(threshold=0.7, model=local_llm),
        
        # Retrieval quality metrics
        ContextualRelevancyMetric(threshold=0.7, model=local_llm),
        ContextualPrecisionMetric(threshold=0.7, model=local_llm),
        
        # Custom correctness metric
        GEval(
            name="Answer Correctness",
            criteria="Evaluate if the actual output provides correct and complete information compared to the expected output.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            threshold=0.7,
            model=local_llm
        )
    ]
    
    assert_test(test_case, metrics)


def test_retrieval_quality():
    """Test the retrieval component separately"""
    
    test_queries = [
        ("waterproof shoes", ["hiking_boots", "warranty"]),
        ("return policy", ["returns"]),
        ("shoe care instructions", ["maintenance"])
    ]
    
    for query, expected_types in test_queries:
        retrieved = rag_app.retrieve_documents(query, k=3)
        retrieved_types = [doc.metadata.get('type', '') for doc in retrieved]
        
        # Check if expected document types are retrieved
        assert any(exp_type in retrieved_types for exp_type in expected_types), \
            f"Failed to retrieve expected documents for query: {query}"


# Advanced metric for conversational quality
def test_conversational_quality():
    """Test if responses are conversational and helpful"""
    
    test_case = LLMTestCase(
        input="My shoes arrived damaged. What should I do?",
        actual_output=rag_app.query("My shoes arrived damaged. What should I do?")[0]
    )
    
    conversational_metric = GEval(
        name="Conversational Quality",
        criteria="""Evaluate if the response is:
        1. Empathetic and understanding
        2. Provides clear next steps
        3. Maintains professional tone
        4. Offers helpful solutions""",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=local_llm
    )
    
    assert_test(test_case, [conversational_metric])


@deepeval.log_hyperparameters
def hyperparameters():
    return {
        "model": "qwen2.5-coder:3b",
        "rag_version": "v1.0",
        "temperature": 0.7,
        "retrieval_k": 3,
        "embedding_dim": 384,
        "chunk_strategy": "semantic",
        "reranking": False
    }


if __name__ == "__main__":
    print("RAG Application with Local LLM - Test Run")
    print("=" * 60)
    
    # Demo queries
    demo_queries = [
        "What kind of running shoes do you have?",
        "Can I return shoes after 45 days?",
        "How do I care for leather shoes?",
        "Do you ship internationally?"
    ]
    
    for query in demo_queries:
        print(f"\nQ: {query}")
        answer, contexts = rag_app.query(query)
        print(f"A: {answer}")
        print(f"Retrieved {len(contexts)} relevant documents")
    
    print("\n" + "=" * 60)
    print("Run full evaluation with:")
    print("deepeval test run test_rag_application.py -v")