"""Test client for eSurge API server."""

import argparse
import asyncio
import json
import sys

import aiohttp


async def test_health(session: aiohttp.ClientSession, base_url: str):
    """Test health check endpoint."""
    print("\n1. Testing health check...")
    async with session.get(f"{base_url}/health") as resp:
        if resp.status == 200:
            data = await resp.json()
            print(f"   ✓ Server status: {data['status']}")
            print(f"   ✓ Models loaded: {list(data['models'].keys())}")
            return True
        else:
            print(f"   ✗ Health check failed: {resp.status}")
            return False


async def test_models(session: aiohttp.ClientSession, base_url: str):
    """Test models endpoint."""
    print("\n2. Testing models endpoint...")
    async with session.get(f"{base_url}/v1/models") as resp:
        if resp.status == 200:
            data = await resp.json()
            print(f"   ✓ Available models: {len(data['data'])}")
            for model in data["data"]:
                print(f"     - {model['id']}")
            return data["data"][0]["id"] if data["data"] else None
        else:
            print(f"   ✗ Models endpoint failed: {resp.status}")
            return None


async def test_chat_completion(session: aiohttp.ClientSession, base_url: str, model_name: str):
    """Test chat completion endpoint."""
    print("\n3. Testing chat completion (non-streaming)...")
    chat_request = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ],
        "max_tokens": 10,
        "temperature": 0.1,
    }

    async with session.post(f"{base_url}/v1/chat/completions", json=chat_request) as resp:
        if resp.status == 200:
            data = await resp.json()
            response_text = data["choices"][0]["message"]["content"]
            print(f"   ✓ Response: {response_text}")
            print(f"   ✓ Tokens used: {data['usage']['total_tokens']}")
            return True
        else:
            print(f"   ✗ Chat completion failed: {resp.status}")
            error = await resp.text()
            print(f"   Error: {error}")
            return False


async def test_streaming(session: aiohttp.ClientSession, base_url: str, model_name: str):
    """Test streaming chat completion."""
    print("\n4. Testing chat completion (streaming)...")
    stream_request = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 30,
        "temperature": 0.5,
        "stream": True,
    }

    async with session.post(f"{base_url}/v1/chat/completions", json=stream_request) as resp:
        if resp.status == 200:
            print("   ✓ Streaming response: ", end="", flush=True)
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        print("\n   ✓ Stream completed")
                        return True
                    try:
                        data = json.loads(data_str)
                        if data.get("choices"):
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass
            return True
        else:
            print(f"   ✗ Streaming failed: {resp.status}")
            return False


async def test_completion(session: aiohttp.ClientSession, base_url: str, model_name: str):
    """Test completion endpoint."""
    print("\n5. Testing completion endpoint...")
    completion_request = {
        "model": model_name,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.1,
    }

    async with session.post(f"{base_url}/v1/completions", json=completion_request) as resp:
        if resp.status == 200:
            data = await resp.json()
            response_text = data["choices"][0]["text"]
            print(f"   ✓ Completion: {response_text}")
            return True
        else:
            print(f"   ✗ Completion failed: {resp.status}")
            return False


async def test_metrics(session: aiohttp.ClientSession, base_url: str):
    """Test metrics endpoint."""
    print("\n6. Testing metrics endpoint...")
    async with session.get(f"{base_url}/metrics") as resp:
        if resp.status == 200:
            data = await resp.json()
            print(f"   ✓ Total requests: {data['total_requests']}")
            print(f"   ✓ Successful requests: {data['successful_requests']}")
            print(f"   ✓ Tokens generated: {data['total_tokens_generated']}")
            print(f"   ✓ Avg tokens/sec: {data['average_tokens_per_second']}")
            return True
        else:
            print(f"   ✗ Metrics failed: {resp.status}")
            return False


async def main(base_url: str):
    """Run all tests."""
    print("=" * 60)
    print("Testing eSurge API Server")
    print(f"Server URL: {base_url}")
    print("=" * 60)

    # Track test results
    results = {}

    async with aiohttp.ClientSession() as session:
        # Test health - required to proceed
        results["health"] = await test_health(session, base_url)
        if not results["health"]:
            print("\n❌ Server health check failed. Is the server running?")
            return False

        # Get model name
        model_name = await test_models(session, base_url)
        if not model_name:
            print("\n❌ No models available")
            return False

        results["models"] = True

        # Test endpoints
        results["chat"] = await test_chat_completion(session, base_url, model_name)
        results["streaming"] = await test_streaming(session, base_url, model_name)
        results["completion"] = await test_completion(session, base_url, model_name)
        results["metrics"] = await test_metrics(session, base_url)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test eSurge API Server")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the eSurge API server")

    args = parser.parse_args()

    # Run tests
    success = asyncio.run(main(args.url))
    sys.exit(0 if success else 1)
