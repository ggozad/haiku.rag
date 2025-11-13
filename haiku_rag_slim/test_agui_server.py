#!/usr/bin/env python3
"""Test script for AG-UI server functionality."""

import asyncio
import json

import httpx


async def test_agui_server():
    """Test the AG-UI server endpoint."""
    base_url = "http://localhost:8000"

    # Test health check
    async with httpx.AsyncClient() as client:
        print("Testing health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Health check response: {response.status_code}")
        print(f"Health check data: {response.json()}")

        # Test AG-UI streaming endpoint
        print("\nTesting AG-UI stream endpoint...")
        request_data = {
            "threadId": "test-thread-1",
            "runId": "test-run-1",
            "state": {"question": "What is pydantic-graph?"},
            "messages": [],
            "config": {},
        }

        print(f"Request data: {json.dumps(request_data, indent=2)}")

        # Send request and stream response
        async with client.stream(
            "POST",
            f"{base_url}/v1/agent/stream",
            json=request_data,
            timeout=120.0,
        ) as response:
            print(f"Response status: {response.status_code}")
            print("Streaming events...\n")

            event_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(event_data)
                        event_type = event.get("type", "UNKNOWN")
                        print(f"Event {event_count}: {event_type}")

                        # Show specific event details
                        if event_type == "RUN_STARTED":
                            print(f"  Thread ID: {event.get('threadId')}")
                            print(f"  Run ID: {event.get('runId')}")
                        elif event_type == "STEP_STARTED":
                            print(f"  Step: {event.get('stepName')}")
                        elif event_type == "ACTIVITY_SNAPSHOT":
                            print(f"  Activity: {event.get('content')}")
                        elif event_type == "STATE_SNAPSHOT":
                            state = event.get("snapshot", {})
                            if "context" in state:
                                context = state["context"]
                                if "sub_questions" in context:
                                    num_questions = len(context["sub_questions"])
                                    print(f"  Sub-questions: {num_questions}")
                        elif event_type == "RUN_FINISHED":
                            result = event.get("result", {})
                            if "title" in result:
                                print(f"  Report Title: {result['title']}")
                        elif event_type == "RUN_ERROR":
                            print(f"  Error: {event.get('message')}")

                        event_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse event: {e}")
                        print(f"Raw line: {line}")

            print(f"\nTotal events received: {event_count}")


if __name__ == "__main__":
    print("AG-UI Server Test")
    print("=" * 50)
    print("Make sure to start the server first with:")
    print("  haiku-rag serve --agui --agui-port 8000")
    print("=" * 50)
    print()

    try:
        asyncio.run(test_agui_server())
    except httpx.ConnectError:
        print("ERROR: Could not connect to server at http://localhost:8000")
        print("Make sure the AG-UI server is running.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
