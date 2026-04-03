"""
Test script to see what thinking tags Qwen3.5-35B-A3B-FP8 actually outputs.
"""
import os
from dotenv import load_dotenv

load_dotenv("/home/lph/Documents/GitHub/AIRAG/.env")

base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
model = os.getenv("OPENAI_COMPATIBLE_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "none")

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url=base_url)

# Test 1: With thinking enabled (non-streaming)
print("=== Test 1: enable_thinking=True (non-streaming) ===")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What does TET Holiday Vietnam mean?"}],
        temperature=0.0,
        max_tokens=5000,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content
    print("reasoning:", repr(reasoning))
    print("---")
    print("content:", repr(content))
except Exception as e:
    print(f"Error: {e}")

print()

# Test 2: With thinking disabled (non-streaming)
print("=== Test 2: enable_thinking=False (non-streaming) ===")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What does TET Holiday Vietnam mean?"}],
        temperature=0.0,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content
    print("reasoning:", repr(reasoning))
    print("---")
    print("content:", repr(content))
except Exception as e:
    print(f"Error: {e}")

print()

# Test 3: Streaming with thinking enabled
print("=== Test 3: Streaming enable_thinking=True ===")
try:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What does TET mean?"}],
        temperature=0.0,
        max_tokens=512,
        stream=True,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            reasoning = getattr(delta, "reasoning", None)
            content = delta.content
            if reasoning:
                print(f"REASONING: {repr(reasoning)}")
            if content:
                print(f"CONTENT: {repr(content)}")
except Exception as e:
    print(f"Error: {e}")
