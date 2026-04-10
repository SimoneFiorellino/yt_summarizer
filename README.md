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

## Installation

```bash
uv sync
```

## Configuration

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Main runtime parameters:

```env
YT_LLM_MODEL=phi3:mini
YT_LLM_TEMPERATURE=0.5
YT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_CHUNK_SIZE=1000
YT_CHUNK_OVERLAP=200
YT_RETRIEVAL_TOP_K=7
```

## Run the App

```bash
uv run src/main.py
```

