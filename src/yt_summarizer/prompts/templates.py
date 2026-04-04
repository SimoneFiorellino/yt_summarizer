"""Prompt templates used by the application."""

from langchain_core.prompts import PromptTemplate


def create_qa_prompt_template():
    """Create the prompt template used for question answering."""
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template,
    )


def summary_prompt():
    template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant tasked with summarizing YouTube video transcripts. Provide a concise, informative summary that captures the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content of the video.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please summarize the following YouTube video transcript:

{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return PromptTemplate(
        input_variables=["transcript"],
        template=template,
    )


def create_summary_prompt():
    """Create the prompt template used for transcript summarization."""
    return summary_prompt()
