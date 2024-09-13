import gradio as gr
import torch
from transformers import pipeline

# Initialize the local pipeline for the model
pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")
# You can choose another model if needed, and adjust parameters as necessary

# Define the response function for local execution
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

    # Use the local pipeline to generate text
    for message in pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
    ):
        token = message['generated_text'][-1]['content']
        response += token
        yield response

# Custom CSS for styling the header and logo
custom_css = """
#header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}

#logo {
    height: 50px; /* Set the height of the logo */
    width: auto; /* Maintain aspect ratio */
    position: absolute; /* Position the logo */
    top: 20px; /* Distance from the top */
    right: 20px; /* Distance from the right */
}
"""

# Gradio interface with additional components and a funny subtitle
demo = gr.ChatInterface(
    respond,
    title="<div id='header'><h1>Bee Chatbot</h1><img id='logo' src='https://raw.githubusercontent.com/atamagnini/mlops-cs553-fall24/main/assets/logo.png'></div>",
    description="Buzzing with answers, but no honey included! 🐝",
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
