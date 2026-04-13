"""Gradio interface for local demos."""

from __future__ import annotations

import json
from urllib import error, request

import gradio as gr

from yt_summarizer.config import AppConfig, load_config


def create_app(config: AppConfig | None = None):
    """Create the Gradio Blocks application."""
    config = config or load_config()
    api_base_url = _api_base_url(config)

    with gr.Blocks() as interface:
        gr.Markdown(
            "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
        )

        video_url = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter the YouTube Video URL",
        )
        ingest_status = gr.Textbox(label="Ingestion Status", lines=2)
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(
            label="Ask a Question About the Video",
            placeholder="Ask your question",
        )
        answer_output = gr.Textbox(label="Answer to Your Question", lines=5)
        ingestion_state = gr.State({"video_id": "", "video_url": ""})

        ingest_btn = gr.Button("Ingest Video")
        summarize_btn = gr.Button("Summarize Video")
        question_btn = gr.Button("Ask a Question")

        ingest_btn.click(
            lambda url, current_state: ingest_video_via_api(
                api_base_url,
                url,
                current_state,
            ),
            inputs=[video_url, ingestion_state],
            outputs=[ingest_status, ingestion_state],
        )
        summarize_btn.click(
            lambda url, current_state: summarize_video_via_api(
                api_base_url,
                url,
                current_state,
            ),
            inputs=[video_url, ingestion_state],
            outputs=[summary_output, ingest_status, ingestion_state],
        )
        question_btn.click(
            lambda url, question, current_state: answer_question_via_api(
                api_base_url,
                url,
                question,
                current_state,
            ),
            inputs=[video_url, question_input, ingestion_state],
            outputs=[answer_output, ingest_status, ingestion_state],
        )

    return interface


def main():
    """Launch the Gradio app."""
    config = load_config()
    create_app(config).launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
    )


def ingest_video_via_api(
    api_base_url: str,
    video_url: str,
    current_state: dict,
) -> tuple[str, dict]:
    """Call the offline ingestion endpoint and persist the returned video_id in UI state."""
    if not video_url:
        return "Please provide a valid YouTube URL.", current_state

    try:
        response = _post_json(
            f"{api_base_url}/ingest_video",
            {"video_url": video_url},
        )
    except RuntimeError as exc:
        return str(exc), current_state

    status = (
        f"Ingested video_id={response['video_id']} | "
        f"language={response['language']} | "
        f"transcript_items={response['transcript_items']} | "
        f"chunk_count={response['chunk_count']} | "
        f"processed_chars={response['processed_chars']}"
    )
    return status, {"video_id": response["video_id"], "video_url": video_url}


def summarize_video_via_api(
    api_base_url: str,
    video_url: str,
    current_state: dict,
) -> tuple[str, str, dict]:
    """Ensure ingestion exists, then call the online summarize endpoint."""
    status_message, resolved_state = _ensure_ingested(
        api_base_url,
        video_url,
        current_state,
    )
    if not resolved_state["video_id"]:
        return status_message, status_message, current_state

    try:
        response = _post_json(
            f"{api_base_url}/summarize",
            {"video_id": resolved_state["video_id"]},
        )
    except RuntimeError as exc:
        message = str(exc)
        return message, status_message, resolved_state

    return response["summary"], status_message, resolved_state


def answer_question_via_api(
    api_base_url: str,
    video_url: str,
    question: str,
    current_state: dict,
) -> tuple[str, str, dict]:
    """Ensure ingestion exists, then call the online QA endpoint."""
    status_message, resolved_state = _ensure_ingested(
        api_base_url,
        video_url,
        current_state,
    )
    if not resolved_state["video_id"]:
        return status_message, status_message, current_state

    try:
        response = _post_json(
            f"{api_base_url}/ask",
            {"video_id": resolved_state["video_id"], "question": question},
        )
    except RuntimeError as exc:
        message = str(exc)
        return message, status_message, resolved_state

    answer = response["answer"]
    sources = response.get("sources", [])
    if sources:
        formatted_sources = "\n".join(
            (
                f"- chunk={source['chunk_id']} | "
                f"{source['source_attribution']} | "
                f"{source['language']} | "
                f"{source['excerpt']}"
            )
            for source in sources
        )
        answer = f"{answer}\n\nSources:\n{formatted_sources}"

    return answer, status_message, resolved_state


def _ensure_ingested(
    api_base_url: str,
    video_url: str,
    current_state: dict,
) -> tuple[str, dict]:
    """Return an existing video_id or trigger offline ingestion first."""
    if current_state["video_id"] and current_state["video_url"] == video_url:
        return (
            f"Reusing existing ingestion for video_id={current_state['video_id']}",
            current_state,
        )

    return ingest_video_via_api(api_base_url, video_url, current_state)


def _post_json(url: str, payload: dict) -> dict:
    """Send a JSON POST request to the backend API."""
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        try:
            parsed = json.loads(detail)
            message = parsed.get("detail", detail)
        except json.JSONDecodeError:
            message = detail or exc.reason
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach the FastAPI backend. Start yt-summarizer-api first."
        ) from exc


def _api_base_url(config: AppConfig) -> str:
    """Build the local API base URL used by the Gradio frontend."""
    api_host = "127.0.0.1" if config.api_host == "0.0.0.0" else config.api_host
    return f"http://{api_host}:{config.api_port}"
