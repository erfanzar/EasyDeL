import json

import requests

# Import after path manipulation

# API endpoint
API_HOST = "http://0.0.0.0:11557"

# Test function definitions
CALCULATOR_FUNCTION = {
    "name": "calculator",
    "description": "Calculate a mathematical expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2 + 2')",
            }
        },
        "required": ["expression"],
    },
}

WEATHER_FUNCTION = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or location"},
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
        },
        "required": ["location"],
    },
}


def test_chat_completion_function_call():
    """Tests the function calling capability"""
    print("Testing function calling...")

    # Define function calling request
    request_data = {
        "model": "LLaMA",
        "messages": [{"role": "user", "content": "What's 25 multiplied by 4?"}],
        "functions": [CALCULATOR_FUNCTION],
        "function_call": {"name": "calculator"},
        "temperature": 0.0,
        "max_tokens": 256,
    }

    # Make the request
    response = requests.post(
        f"{API_HOST}/v1/chat/completions",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    # Validate the response
    if response.status_code == 200:
        response_data = response.json()
        print(f"Response received: {json.dumps(response_data, indent=2)}")

        # Check if function was called
        message = response_data["choices"][0]["message"]
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]
            print(f"Function called: {function_name}")
            print(f"Arguments: {arguments}")
            print("✓ Function calling test passed")
        else:
            print("✗ Function was not called in response")
    else:
        print(f"✗ Request failed with status code {response.status_code}")
        print(response.text)


def test_chat_completion_tool_call():
    """Tests the tool calling capability (functions with tools interface)"""
    print("\nTesting tool calling...")

    # Define tool calling request
    request_data = {
        "model": "LLaMA",
        "messages": [{"role": "user", "content": "What's the weather like in New York?"}],
        "tools": [{"type": "function", "function": WEATHER_FUNCTION}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "temperature": 0.0,
        "max_tokens": 256,
    }

    # Make the request
    response = requests.post(
        f"{API_HOST}/v1/chat/completions",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    # Validate the response
    if response.status_code == 200:
        response_data = response.json()
        print(f"Response received: {json.dumps(response_data, indent=2)}")

        # Check if function was called via tool interface
        message = response_data["choices"][0]["message"]
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]
            print(f"Tool called: {function_name}")
            print(f"Arguments: {arguments}")
            print("✓ Tool calling test passed")
        else:
            print("✗ Tool was not called in response")
    else:
        print(f"✗ Request failed with status code {response.status_code}")
        print(response.text)


def test_streaming_function_call():
    """Tests streaming mode with function calling"""
    print("\nTesting streaming function call...")

    # Define function calling request with streaming
    request_data = {
        "model": "LLaMA",
        "messages": [{"role": "user", "content": "What's the square root of 144?"}],
        "functions": [CALCULATOR_FUNCTION],
        "function_call": "auto",
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": True,
    }

    # Make the streaming request
    with requests.post(
        f"{API_HOST}/v1/chat/completions",
        json=request_data,
        headers={"Content-Type": "application/json"},
        stream=True,
    ) as response:
        if response.status_code == 200:
            function_call_detected = False
            print("Streaming response chunks:")

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]

                        if delta.get("function_call"):
                            function_call_detected = True
                            print(f"Function call detected in streaming: {delta['function_call']}")
                    except json.JSONDecodeError:
                        print(f"Could not parse: {line}")

            if function_call_detected:
                print("✓ Streaming function call test passed")
            else:
                print("✗ Function call not detected in streaming response")
        else:
            print(f"✗ Streaming request failed with status code {response.status_code}")
            print(response.text)


def test_completion_endpoint():
    """Tests the /v1/completions endpoint"""
    print("\nTesting completions endpoint...")

    request_data = {
        "model": "LLaMA",
        "prompt": "Write a haiku about programming:",
        "max_tokens": 64,
        "temperature": 0.7,
    }

    response = requests.post(
        f"{API_HOST}/v1/completions",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        response_data = response.json()
        print(f"Completion response: {json.dumps(response_data, indent=2)}")

        # Verify the response format
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            text = response_data["choices"][0]["text"]
            print(f"Generated text: {text}")
            print("✓ Completion test passed")
        else:
            print("✗ No choices in completion response")
    else:
        print(f"✗ Completion request failed with status code {response.status_code}")
        print(response.text)


def test_streaming_completion():
    """Tests streaming mode with the completions endpoint"""
    print("\nTesting streaming completion...")

    request_data = {
        "model": "LLaMA",
        "prompt": "Count from 1 to 5:",
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
    }

    with requests.post(
        f"{API_HOST}/v1/completions",
        json=request_data,
        headers={"Content-Type": "application/json"},
        stream=True,
    ) as response:
        if response.status_code == 200:
            print("Streaming completion chunks:")
            accumulated_text = ""

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if chunk["choices"] and len(chunk["choices"]) > 0:
                            text = chunk["choices"][0]["text"]
                            accumulated_text += text
                            print(f"Chunk: {text}")
                    except json.JSONDecodeError:
                        print(f"Could not parse: {line}")

            print(f"Complete text: {accumulated_text}")
            print("✓ Streaming completion test passed")
        else:
            print(f"✗ Streaming completion request failed with status code {response.status_code}")
            print(response.text)


def test_client_usage():
    """Tests usage of the API client"""
    print("\nTesting API client usage...")

    try:
        from easydel.inference.vinference.api_server import (
            ChatCompletionRequest,
            ChatMessage,
            CompletionRequest,
            vInferenceClient,
        )

        client = vInferenceClient(API_HOST)

        # Test chat completion
        chat_request = ChatCompletionRequest(
            model="LLaMA",
            messages=[ChatMessage(role="user", content="Hello, who are you?")],
            max_tokens=32,
        )

        print("Testing chat client...")
        chat_response = next(client.chat.create_chat_completion(chat_request))
        print(f"Chat response: {chat_response.model_dump()}")

        # Test text completion
        completion_request = CompletionRequest(model="LLaMA", prompt="Complete this: The sky is", max_tokens=16)

        print("\nTesting completion client...")
        completion_response = next(client.completions.create_completion(completion_request))
        print(f"Completion response: {completion_response.model_dump()}")

        print("✓ Client usage test passed")
    except Exception as e:
        print(f"✗ Client test failed: {e!s}")


if __name__ == "__main__":
    print("=== vInference API Function and Completions Tests ===")
    print(f"Testing against API at {API_HOST}")
    print("Make sure the API server is running before executing these tests")

    # Run all tests
    test_chat_completion_function_call()
    test_chat_completion_tool_call()
    test_streaming_function_call()
    test_completion_endpoint()
    test_streaming_completion()
    test_client_usage()

    print("\nAll tests completed!")
