#!/bin/bash
# DeepEval Docker Runner Script
# Usage: ./scripts/docker-run.sh [command]
#
# Commands:
#   start       - Start Ollama server (CPU mode)
#   start-gpu   - Start Ollama server with GPU support
#   pull-model  - Pull the configured model
#   test        - Run DeepEval tests
#   dev         - Start interactive development shell
#   stop        - Stop all services
#   logs        - Show Ollama logs
#   status      - Check service status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default model if not set
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2}

case "${1:-help}" in
    start)
        echo "Starting Ollama server (CPU mode)..."
        docker compose --profile cpu up -d ollama-cpu
        echo "Waiting for Ollama to be ready..."
        sleep 5
        echo "Ollama is running at http://localhost:${OLLAMA_HOST_PORT:-11434}"
        ;;

    start-gpu)
        echo "Starting Ollama server with GPU support..."
        docker compose --profile gpu up -d ollama
        echo "Waiting for Ollama to be ready..."
        sleep 5
        echo "Ollama is running at http://localhost:${OLLAMA_HOST_PORT:-11434}"
        ;;

    pull-model)
        echo "Pulling model: $OLLAMA_MODEL..."
        docker compose exec ollama-cpu ollama pull "$OLLAMA_MODEL" 2>/dev/null || \
        docker compose exec ollama ollama pull "$OLLAMA_MODEL" 2>/dev/null || \
        echo "Make sure Ollama is running first: ./scripts/docker-run.sh start"
        ;;

    test)
        echo "Running DeepEval tests..."
        docker compose --profile cpu --profile test up --build deepeval
        ;;

    dev)
        echo "Starting development shell..."
        docker compose --profile cpu --profile dev run --rm deepeval-dev
        ;;

    stop)
        echo "Stopping all services..."
        docker compose --profile cpu --profile gpu --profile test --profile dev down
        ;;

    logs)
        docker compose logs -f ollama-cpu 2>/dev/null || \
        docker compose logs -f ollama 2>/dev/null
        ;;

    status)
        echo "Service Status:"
        docker compose ps
        echo ""
        echo "Checking Ollama connection..."
        curl -s "http://localhost:${OLLAMA_HOST_PORT:-11434}/api/tags" | head -c 200 || \
        echo "Ollama not responding"
        ;;

    help|*)
        echo "DeepEval Docker Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start       Start Ollama server (CPU mode)"
        echo "  start-gpu   Start Ollama server with GPU support"
        echo "  pull-model  Pull the configured model ($OLLAMA_MODEL)"
        echo "  test        Run DeepEval tests in Docker"
        echo "  dev         Start interactive development shell"
        echo "  stop        Stop all services"
        echo "  logs        Show Ollama logs"
        echo "  status      Check service status"
        echo ""
        echo "Environment Variables (set in .env):"
        echo "  OLLAMA_BASE_URL   - Ollama server URL"
        echo "  OLLAMA_MODEL      - Model to use (default: llama3.2)"
        echo "  OLLAMA_HOST_PORT  - Host port for Ollama (default: 11434)"
        ;;
esac
