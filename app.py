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
    uploaded_file=None,
    dropdown_option="Default",
    checkbox_value=False,
    color="white"
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Process the uploaded file if present
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        # Use the content as needed; for now, just append it to the system message
        messages.append({"role": "user", "content": f"User uploaded a file: {file_content}"})

    # Process dropdown option
    messages.append({"role": "user", "content": f"User selected option: {dropdown_option}"})

    # Process checkbox value
    if checkbox_value:
        messages.append({"role": "user", "content": "User checked the checkbox."})

    # Process color picker
    messages.append({"role": "user", "content": f"User selected color: {color}"})
    
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
# Create the Gradio Chat Interface with additional components
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.File(label="Upload a File"),  # File uploader
        gr.Image(type="pil", label="Display Image"),  # Image viewer
        gr.Dropdown(["Option 1", "Option 2", "Option 3"], label="Choose an option", value="Option 1"),  # Dropdown menu
        gr.Checkbox(label="Check me!"),  # Checkbox
        gr.ColorPicker(label="Choose a color"),  # Color picker
        gr.TextArea(label="Additional notes", placeholder="Type any additional notes here..."),  # Text area
        gr.Button(value="Custom Button", label="Press me!")  # Custom button
    ],
)


if __name__ == "__main__":
    demo.launch()