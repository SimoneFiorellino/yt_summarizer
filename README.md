# yt-summarizer

A small Python app that fetches a YouTube transcript, generates a summary, and lets you ask questions about the video through a simple Gradio UI.

<p align="center">
  <img src="assets/app.png" alt="App screenshot" width="800"/>
</p>

## Features

- Fetches English transcripts from YouTube
- Summarizes the video content
- Supports question answering over the transcript
- Provides a local web interface with Gradio

## Libraries Used

- `gradio` for the web UI
- `youtube-transcript-api` for retrieving video transcripts
- `langchain` for prompt and chain orchestration
- `langchain-ollama` for the LLM integration
- `langchain-huggingface` and `sentence-transformers` for embeddings
- `faiss-cpu` for vector search over transcript chunks

## Project Structure

```text
src/
  main.py
  yt_summarizer/
    cli.py
    workflow.py
    llm/
    prompts/
    retrieval/
    transcript/
```

## Installation

```bash
uv sync
```

## Run the App

```bash
uv run src/main.py
```

The app launches locally on port `7865`.
