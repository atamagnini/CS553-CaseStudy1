import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the InferenceClient
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Define the response function
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Custom CSS for styling the Submit button and other elements
custom_css = """
body {
    background-color: #f0f8ff; /* Light blue background for the whole interface */
}

.gradio-container {
    border-radius: 10px; /* Rounded corners for the whole Gradio interface */
    padding: 20px;
}

#header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px; /* Adds space below the header */
}

#logo {
    height: 50px; /* Logo size */
}

/* Style the Submit button */
.gr-button {
    background-color: #4CAF50 !important; /* Green background for the button */
    color: white !important; /* White text color */
    font-family: 'Arial', sans-serif !important; /* Custom font for the button text */
    font-size: 16px !important; /* Font size */
    padding: 10px 20px !important; /* Padding for the button */
    border-radius: 5px !important; /* Rounded corners for the button */
    border: none !important; /* Remove border */
    cursor: pointer !important; /* Change cursor on hover */
}

/* Style the Submit button on hover */
.gr-button:hover {
    background-color: #45a049 !important; /* Darker green on hover */
}
"""

# Gradio interface with additional components
demo = gr.ChatInterface(
    respond,
    title="<div id='header'><h1>Bee Chatbot</h1><img id='logo' src='https://raw.githubusercontent.com/atamagnini/mlops-cs553-fall24/main/assets/logo.png'></div>",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    css=custom_css
)

if __name__ == "__main__":
    demo.launch()
