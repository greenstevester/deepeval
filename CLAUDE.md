# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Setup and Installation
```bash
# Install with Poetry (preferred)
poetry install

# Install development dependencies
poetry install --with dev

# Or install with pip (if Poetry is not available)
pip install -e .
```

### Running Tests
```bash
# Run all tests
deepeval test run

# Run a specific test file
deepeval test run test_<filename>.py

# Run tests in parallel
deepeval test run test_<filename>.py -n 4

# Run tests excluding those marked with skip_test
pytest -m 'not skip_test'

# Run specific test
pytest tests/test_metrics/test_answer_relevancy.py::test_answer_relevancy -v
```

### Code Formatting and Linting
```bash
# Format code with Black (line length 80)
black deepeval/ --line-length 80

# Check formatting without making changes
black deepeval/ --line-length 80 --check
```

### Building and Publishing
```bash
# Build package
poetry build

# Publish to PyPI (requires credentials)
poetry publish
```

## High-Level Architecture

### Core Components

1. **Metrics** (`deepeval/metrics/`)
   - Base metric class in `base_metric.py` that all metrics inherit from
   - Each metric has its own directory with:
     - Main implementation file (e.g., `answer_relevancy.py`)
     - Schema definitions (`schema.py`)
     - Prompt templates (`template.py`)
   - Special metric categories:
     - RAG metrics: answer_relevancy, faithfulness, contextual_*
     - Agentic metrics: task_completion, tool_correctness
     - Multimodal metrics: in `multimodal_metrics/`
     - Conversational metrics: conversation_completeness, turn_relevancy

2. **Test Cases** (`deepeval/test_case/`)
   - `LLMTestCase`: Standard test case for text-based LLM evaluation
   - `ConversationalTestCase`: For multi-turn conversations
   - `ArenaTestCase`: For comparative evaluations
   - `MLLMTestCase`: For multimodal evaluations

3. **Models** (`deepeval/models/`)
   - LLM providers in `llms/`: OpenAI, Anthropic, Gemini, local models, etc.
   - Embedding models in `embedding_models/`
   - Multimodal LLMs in `mlllms/`
   - All models inherit from base classes that define standard interfaces

4. **Evaluation System** (`deepeval/evaluate/`)
   - `evaluate.py`: Main evaluation logic for running metrics on test cases
   - `assert_test()`: Pytest integration for test assertions
   - `compare.py`: For comparing multiple evaluation runs

5. **Tracing** (`deepeval/tracing/`)
   - Component-level evaluation support via `@observe` decorator
   - OpenTelemetry integration for distributed tracing
   - Offline evaluation of traced components

6. **CLI** (`deepeval/cli/`)
   - Main entry point in `main.py`
   - Test runner in `test.py`
   - Commands: `deepeval test run`, `deepeval login`, etc.

7. **Integrations** (`deepeval/integrations/`)
   - Framework integrations: LangChain, LlamaIndex, HuggingFace
   - Each integration provides callbacks/handlers for seamless evaluation

8. **Synthesizer** (`deepeval/synthesizer/`)
   - Generates synthetic test data from various sources
   - Document chunking and context generation

9. **Benchmarks** (`deepeval/benchmarks/`)
   - Standard LLM benchmarks: MMLU, HumanEval, GSM8K, etc.
   - Each benchmark has task definitions and evaluation logic

### Key Design Patterns

1. **Metric Interface**: All metrics implement `BaseMetric` with:
   - `measure()`: Synchronous evaluation
   - `a_measure()`: Asynchronous evaluation
   - Score normalization (0-1 range)
   - Detailed reasoning/explanation

2. **Modular LLM Support**: Any LLM can be used by implementing the base model interface
   - Supports both sync and async generation
   - Handles retries and error handling

3. **Plugin System**: Pytest plugin in `deepeval/plugins/` for test discovery and execution

4. **Confident AI Integration**: Optional cloud platform integration for:
   - Test result visualization
   - Dataset management
   - Metric fine-tuning

### Environment Variables

- `OPENAI_API_KEY`: For OpenAI model usage
- `DEEPEVAL_TELEMETRY_OPT_OUT`: Disable telemetry
- `DEEPEVAL_UPDATE_WARNING_OPT_IN`: Enable update warnings
- `DEEPEVAL_FILE_SYSTEM`: Set to "READ_ONLY" for read-only environments

### Testing Approach

- Tests are in `tests/` directory, organized by component
- Integration tests in `tests/integrations/`
- Use pytest markers: `@pytest.mark.skip_test` to skip tests
- Async tests supported via pytest-asyncio