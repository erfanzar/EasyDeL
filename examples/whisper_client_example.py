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
Example client for the vWhisper FastAPI server.

This script demonstrates how to send requests to the vWhisper server
using the requests library.
"""

import argparse
import json
import sys

import requests


def transcribe_audio(audio_file, server_url, language=None, response_format="json", timestamps=False):
    """
    Transcribe audio using the vWhisper server.

    Args:
        audio_file: Path to the audio file
        server_url: URL of the vWhisper server
        language: Language code (optional)
        response_format: Response format (json, text, srt, verbose_json, vtt)
        timestamps: Whether to include timestamps

    Returns:
        The server response as a JSON object
    """
    endpoint = f"{server_url}/v1/audio/transcriptions"

    # Prepare form data
    data = {
        "model": "whisper-large-v3-turbo",  # This is ignored by the server but required by the API
        "response_format": response_format,
    }

    if language:
        data["language"] = language

    if timestamps:
        data["timestamp_granularities"] = ["word"]

    # Prepare file
    files = {"file": open(audio_file, "rb")}

    # Send request
    response = requests.post(endpoint, files=files, data=data)

    # Check if request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def translate_audio(audio_file, server_url, response_format="json", timestamps=False):
    """
    Translate audio to English using the vWhisper server.

    Args:
        audio_file: Path to the audio file
        server_url: URL of the vWhisper server
        response_format: Response format (json, text, srt, verbose_json, vtt)
        timestamps: Whether to include timestamps

    Returns:
        The server response as a JSON object
    """
    endpoint = f"{server_url}/v1/audio/translations"

    # Prepare form data
    data = {
        "model": "whisper-large-v3-turbo",  # This is ignored by the server but required by the API
        "response_format": response_format,
    }

    if timestamps:
        data["timestamp_granularities"] = ["word"]

    # Prepare file
    files = {"file": open(audio_file, "rb")}

    # Send request
    response = requests.post(endpoint, files=files, data=data)

    # Check if request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def main():
    parser = argparse.ArgumentParser(description="vWhisper client example")

    parser.add_argument("audio_file", type=str, help="Path to the audio file to transcribe/translate")

    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="URL of the vWhisper server",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Mode: transcribe or translate",
    )

    parser.add_argument("--language", type=str, help="Language code (only for transcription)")

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text", "srt", "verbose_json", "vtt"],
        default="json",
        help="Response format",
    )

    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in the response")

    args = parser.parse_args()

    # Check if audio file exists
    try:
        with open(args.audio_file, "rb"):
            pass
    except FileNotFoundError:
        print(f"Error: Audio file {args.audio_file} not found")
        sys.exit(1)

    # Process audio
    if args.mode == "transcribe":
        result = transcribe_audio(args.audio_file, args.server, args.language, args.format, args.timestamps)
    else:  # translate
        result = translate_audio(args.audio_file, args.server, args.format, args.timestamps)

    # Print result
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
