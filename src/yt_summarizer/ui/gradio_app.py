"""Gradio interface for local demos."""

import gradio as gr

from yt_summarizer.config import load_config
from yt_summarizer.workflow import answer_question, summarize_video


def create_app():
    """Create the Gradio Blocks application."""
    with gr.Blocks() as interface:
        gr.Markdown(
            "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
        )

        video_url = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter the YouTube Video URL",
        )
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(
            label="Ask a Question About the Video",
            placeholder="Ask your question",
        )
        answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

        summarize_btn = gr.Button("Summarize Video")
        question_btn = gr.Button("Ask a Question")

        summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
        question_btn.click(
            answer_question,
            inputs=[video_url, question_input],
            outputs=answer_output,
        )

    return interface


def main():
    """Launch the Gradio app."""
    config = load_config()
    create_app().launch(server_name=config.gradio_host, server_port=config.gradio_port)
