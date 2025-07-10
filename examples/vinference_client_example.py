#!/usr/bin/env python3
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example client for EasyDeL vInference API Server.

This script demonstrates how to:
1. Send text-only requests to the vInference API
2. Send multimodal (text+image) requests
3. Stream responses from the API
4. Count tokens
"""

import argparse
import base64
import json
import time
from typing import Any

import requests


def text_completion(
    api_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Send a text completion request to the vInference API.

    Args:
        api_url: Base URL of the API server
        model_id: ID of the model to use
        prompt: Text prompt for completion
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-1)
        top_p: Top-p sampling parameter (0-1)
        stream: Whether to stream the response

    Returns:
        API response as a dictionary
    """
    endpoint = f"{api_url}/v1/chat/completions"

    # Prepare the request payload
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # Send the request
    if not stream:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
    else:
        # Handle streaming response
        response = requests.post(endpoint, json=payload, stream=True)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}

        # Process the streaming response
        collected_chunks = []
        collected_messages = []

        # Print the streaming response and collect chunks
        print("\nStreaming response:")
        print("-" * 40)

        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode("utf-8")
                if chunk_str.startswith("data: "):
                    chunk_data = chunk_str[6:]
                    if chunk_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(chunk_data)
                        collected_chunks.append(json_data)
                        delta = json_data["choices"][0]["delta"]
                        if delta.get("content"):
                            collected_messages.append(delta["content"])
                            print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        print("\n" + "-" * 40)

        # Construct the full response
        result = {
            "choices": [{"message": {"role": "assistant", "content": "".join(collected_messages)}}],
            "chunks": collected_chunks,
        }

        return result


def multimodal_completion(
    api_url: str,
    model_id: str,
    text_prompt: str,
    image_path: str | None = None,
    image_url: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Send a multimodal (text+image) completion request to the vInference API.

    Args:
        api_url: Base URL of the API server
        model_id: ID of the model to use
        text_prompt: Text prompt for the image
        image_path: Local path to an image file (optional)
        image_url: URL to an image (optional)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-1)
        stream: Whether to stream the response

    Returns:
        API response as a dictionary
    """
    if image_path is None and image_url is None:
        raise ValueError("Either image_path or image_url must be provided")

    endpoint = f"{api_url}/v1/chat/completions"

    # Prepare content list
    content = []

    # Add image to content
    if image_url:
        # Use image URL directly
        content.append({"type": "image", "image": image_url})
    elif image_path:
        # Encode image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_image}"})

    # Add text to content
    content.append({"type": "text", "text": text_prompt})

    # Prepare the request payload
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # Send the request
    if not stream:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
    else:
        # Handle streaming response
        response = requests.post(endpoint, json=payload, stream=True)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}

        # Process the streaming response
        collected_chunks = []
        collected_messages = []

        # Print the streaming response and collect chunks
        print("\nStreaming response:")
        print("-" * 40)

        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode("utf-8")
                if chunk_str.startswith("data: "):
                    chunk_data = chunk_str[6:]
                    if chunk_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(chunk_data)
                        collected_chunks.append(json_data)
                        delta = json_data["choices"][0]["delta"]
                        if delta.get("content"):
                            collected_messages.append(delta["content"])
                            print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        print("\n" + "-" * 40)

        # Construct the full response
        result = {
            "choices": [{"message": {"role": "assistant", "content": "".join(collected_messages)}}],
            "chunks": collected_chunks,
        }

        return result


def count_tokens(api_url: str, model_id: str, text: str) -> dict[str, Any]:
    """
    Count tokens in a text string.

    Args:
        api_url: Base URL of the API server
        model_id: ID of the model to use
        text: Text to count tokens for

    Returns:
        API response with token count
    """
    endpoint = f"{api_url}/v1/count_tokens"

    payload = {"model": model_id, "conversation": text}

    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}


def get_available_models(api_url: str) -> dict[str, Any]:
    """
    Get available models from the API.

    Args:
        api_url: Base URL of the API server

    Returns:
        List of available models
    """
    endpoint = f"{api_url}/v1/models"

    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}


def main():
    parser = argparse.ArgumentParser(description="EasyDeL vInference API Client Example")

    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vInference API server",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Text completion command
    text_parser = subparsers.add_parser("text", help="Text-only completion")
    text_parser.add_argument("--model", type=str, required=True, help="Model ID")
    text_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    text_parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens to generate")
    text_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    text_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    text_parser.add_argument("--stream", action="store_true", help="Stream the response")

    # Multimodal completion command
    mm_parser = subparsers.add_parser("multimodal", help="Multimodal (text+image) completion")
    mm_parser.add_argument("--model", type=str, required=True, help="Model ID")
    mm_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    mm_parser.add_argument("--image_path", type=str, help="Path to local image file")
    mm_parser.add_argument("--image_url", type=str, help="URL to image")
    mm_parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens to generate")
    mm_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    mm_parser.add_argument("--stream", action="store_true", help="Stream the response")

    # Token counting command
    token_parser = subparsers.add_parser("count", help="Count tokens in text")
    token_parser.add_argument("--model", type=str, required=True, help="Model ID")
    token_parser.add_argument("--text", type=str, required=True, help="Text to count tokens for")

    # List models command
    subparsers.add_parser("models", help="List available models")

    args = parser.parse_args()

    # Execute the selected command
    if args.command == "text":
        start_time = time.time()
        result = text_completion(
            api_url=args.api_url,
            model_id=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=args.stream,
        )
        elapsed = time.time() - start_time

        if not args.stream and "error" not in result:
            print("\nResponse:")
            print("-" * 40)
            print(result["choices"][0]["message"]["content"])
            print("-" * 40)
            if "usage" in result:
                print(
                    f"Tokens: {result['usage']['prompt_tokens']} (prompt) + "
                    f"{result['usage']['completion_tokens']} (completion) = "
                    f"{result['usage']['total_tokens']} (total)"
                )
            print(f"Time: {elapsed:.2f} seconds")

    elif args.command == "multimodal":
        if not args.image_path and not args.image_url:
            print("Error: Either --image_path or --image_url must be provided")
            return

        start_time = time.time()
        result = multimodal_completion(
            api_url=args.api_url,
            model_id=args.model,
            text_prompt=args.prompt,
            image_path=args.image_path,
            image_url=args.image_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )
        elapsed = time.time() - start_time

        if not args.stream and "error" not in result:
            print("\nResponse:")
            print("-" * 40)
            print(result["choices"][0]["message"]["content"])
            print("-" * 40)
            if "usage" in result:
                print(
                    f"Tokens: {result['usage']['prompt_tokens']} (prompt) + "
                    f"{result['usage']['completion_tokens']} (completion) = "
                    f"{result['usage']['total_tokens']} (total)"
                )
            print(f"Time: {elapsed:.2f} seconds")

    elif args.command == "count":
        result = count_tokens(api_url=args.api_url, model_id=args.model, text=args.text)
        print(f"Token count: {result.get('count', 'Error')}")

    elif args.command == "models":
        result = get_available_models(args.api_url)
        if "data" in result:
            print("Available models:")
            for model in result["data"]:
                print(f"- {model['id']}")
        else:
            print("Error retrieving models")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
