import pytest
from app import respond  # Import chatbot function

def test_chatbot_response():
    # Set up the input parameters for chatbot
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
    assert "Hello" not in response, "Response should not simply echo the input"
