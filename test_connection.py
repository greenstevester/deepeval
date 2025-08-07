"""
Test connection to local LLM server
"""

import requests
import json

# Test basic connectivity
base_url = "http://10.0.0.125:11434"

print("Testing connection to local LLM server...")
print(f"Base URL: {base_url}")

# Test 1: Basic connectivity
try:
    response = requests.get(f"{base_url}/", timeout=5)
    print(f"✓ Server is reachable (status: {response.status_code})")
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to server. Please check:")
    print("  1. Is the server running?")
    print("  2. Is the IP address correct?")
    print("  3. Is port 11434 open?")
    exit(1)
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# Test 2: Check API endpoints
print("\nChecking API endpoints...")
endpoints = ["/v1/models", "/v1/chat/completions", "/api/tags"]

for endpoint in endpoints:
    try:
        if endpoint == "/v1/chat/completions":
            # POST request for chat completions
            data = {
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.1
            }
            response = requests.post(
                f"{base_url}{endpoint}", 
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
        else:
            # GET request for other endpoints
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
        
        print(f"  {endpoint}: {response.status_code}")
        
        if response.status_code == 200 and endpoint == "/api/tags":
            # List available models if using Ollama
            models = response.json().get("models", [])
            if models:
                print("\n  Available models:")
                for model in models:
                    print(f"    - {model.get('name', 'unknown')}")
                    
    except Exception as e:
        print(f"  {endpoint}: Error - {e}")

print("\nIf you're using Ollama, make sure to:")
print("  1. Start Ollama: ollama serve")
print("  2. Pull a model: ollama pull llama3.2")
print("  3. Verify it's running: curl http://10.0.0.125:11434/api/tags")