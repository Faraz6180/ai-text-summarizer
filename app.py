import gradio as gr
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize(text, style, length):
    if not text or text.strip() == "":
        return "Please paste some text first."

    prompts = {
        "Bullet Points": f"Summarize into bullet points:\n\n{text}",
        "Paragraph": f"Summarize into one paragraph:\n\n{text}",
        "Executive Summary": f"Write a professional executive summary:\n\n{text}",
        "Simple English": f"Explain in simple English:\n\n{text}"
    }

    prompt = prompts[style] + f"\n\nSummary length approx {length} words."

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    return response.choices[0].message.content


def upload_file(file):
    if file is None:
        return ""
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI Text Summarizer")
    gr.Markdown("Powered by LLaMA 3 + Groq API")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload TXT File")
            text_input = gr.Textbox(lines=12, label="Paste Text")

            style = gr.Dropdown(
                ["Bullet Points", "Paragraph", "Executive Summary", "Simple English"],
                value="Bullet Points",
                label="Summary Style"
            )

            length = gr.Slider(50, 300, value=120, label="Summary Length")

            summarize_btn = gr.Button("Generate Summary")

        with gr.Column():
            output = gr.Textbox(lines=15, label="Summary Output")

    file_input.change(upload_file, file_input, text_input)
    summarize_btn.click(summarize, [text_input, style, length], output)

demo.launch()
