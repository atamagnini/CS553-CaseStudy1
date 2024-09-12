import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


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

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

# Custom CSS for changing the color scheme and adding a logo
custom_css = """
#component-1 {
    background-color: #f0f8ff; /* Light blue background */
    border-radius: 10px; /* Rounded corners for the input components */
    padding: 10px;
}

#component-1 h1 {
    display: inline-block;
    margin: 0;
}

#header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

#logo {
    height: 50px;
}
"""

# Gradio interface with additional components
demo = gr.ChatInterface(
    respond,
    title="<div id='header'><h1>My Enhanced Chatbot</h1><img id='logo' src='https://github.com/atamagnini/mlops-cs553-fall24/blob/main/assets/logo.png'></div>"
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
