import sys
import os
import pytest

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import respond  # Import your chatbot function

def test_chatbot_response():
    # Set up the input parameters for your chatbot
    message = "Hello"
    history = []  # Empty history for a new conversation
    system_message = "You are a friendly chatbot."
    max_tokens = 50
    temperature = 0.7
    top_p = 0.95

    # Run the chatbot function
    response_generator = respond(message, history, system_message, max_tokens, temperature, top_p)

    # Collect the response from the generator
    response = next(response_generator)

    # Assertions to check that the response is not empty and seems reasonable
    assert response is not None, "Response should not be None"
    assert len(response) > 0, "Response should not be empty"
    assert len(response) > len(message), "Response should contain more than just the input"
    assert response.lower() != "hello", "Response should not be just 'Hello'"
