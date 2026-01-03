#!/bin/bash
# DeepEval Docker Runner Script
# Usage: ./scripts/docker-run.sh [command]
#
# Requires external Ollama server (set OLLAMA_BASE_URL in .env)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Defaults
OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5-coder:7b}

case "${1:-help}" in
    test)
        echo "Running DeepEval tests..."
        echo "Using Ollama at: $OLLAMA_BASE_URL"
        echo "Using model: $OLLAMA_MODEL"
        docker compose --profile test run --rm deepeval
        ;;

    dev)
        echo "Starting development shell..."
        docker compose --profile dev run --rm deepeval-dev
        ;;

    build)
        echo "Building DeepEval image..."
        docker compose build
        ;;

    stop)
        echo "Stopping all services..."
        docker compose down
        ;;

    clean)
        echo "Cleaning up Docker resources..."
        docker compose down --volumes --remove-orphans
        docker image prune -f
        ;;

    help|*)
        echo "DeepEval Docker Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  test    Run DeepEval tests in Docker"
        echo "  dev     Start interactive development shell"
        echo "  build   Build the DeepEval Docker image"
        echo "  stop    Stop all services"
        echo "  clean   Clean up Docker resources"
        echo ""
        echo "Configuration (set in .env or environment):"
        echo "  OLLAMA_BASE_URL  Ollama server URL (current: $OLLAMA_BASE_URL)"
        echo "  OLLAMA_MODEL     Model to use (current: $OLLAMA_MODEL)"
        ;;
esac
