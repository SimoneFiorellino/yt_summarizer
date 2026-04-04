"""LangChain chain builders."""

from langchain_classic.chains import LLMChain


def create_qa_chain(llm, prompt_template, verbose=True):
    """Build the chain used for question answering."""
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def create_summary_chain(llm, prompt, verbose=True):
    """Build the chain used for transcript summarization."""
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)
