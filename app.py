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
    title="<div id='header'><h1>Bee Chatbot</h1><h3 style='color:gray;'>Making honey out of conversations!</h3><img id='logo' src='https://raw.githubusercontent.com/atamagnini/mlops-cs553-fall24/main/assets/logo.png'></div>",
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
