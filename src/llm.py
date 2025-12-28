"""
LLM module for RAG operations using local Phi-3-mini model.
Handles query expansion, intent classification, and response generation.
"""

import logging
from typing import List, Generator

logger = logging.getLogger(__name__)

# Phi-3-mini has 4k context, but we need to leave room for generation
MAX_CONTEXT_CHARS = 2000  # Limit context to ~500 tokens


def get_llm():
    """Get the local LLM instance."""
    from src.local_llm import LocalLLM
    return LocalLLM()


def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to fit within token limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def expand_query(query: str) -> str:
    """
    Expand a simple user query into a more semantically rich query for vector search.
    Uses the local LLM for query expansion.
    """
    system_prompt = "Rewrite the query with synonyms and related terms. Output ONLY the expanded query."
    user_prompt = f"Expand: {query}"

    try:
        llm = get_llm()
        expanded = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=200,  # Total sequence length
            temperature=0.3
        )
        expanded = expanded.strip()
        if expanded:
            return expanded
        return query
    except Exception as e:
        logger.error(f"Error expanding query: {e}")
        return query


def classify_intent(query: str) -> str:
    """
    Classify the user's intent into categories: summary, author, flowchart, or rag.
    """
    system_prompt = "Classify as: summary, author, flowchart, or rag. Output ONLY the category."
    user_prompt = f"Query: {query}"

    try:
        llm = get_llm()
        intent = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=150,  # Total sequence length (not just output)
            temperature=0.1
        )
        intent = intent.strip().lower()
        # Extract just the category word
        for cat in ['summary', 'author', 'flowchart', 'rag']:
            if cat in intent:
                return cat
        return 'rag'
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return 'rag'


def generate_response(
    query: str,
    context_chunks: List[dict],
    intent: str
) -> Generator[str, None, None]:
    """
    Generate a response based on the query, context, and intent.
    Yields tokens for streaming output.
    """
    # Build context with truncation
    context_parts = []
    total_len = 0
    for chunk in context_chunks:
        chunk_text = f"[{chunk['source']} p{chunk['page']}]: {chunk['text'][:500]}"
        if total_len + len(chunk_text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk_text)
        total_len += len(chunk_text)
    
    context_text = "\n".join(context_parts) if context_parts else "No context available."

    if intent == 'flowchart':
        system_prompt = "Generate Mermaid.js diagram code only."
        user_prompt = query

    elif intent == 'summary':
        system_prompt = f"Context: {context_text}\n\nSummarize in 2-3 sentences."
        user_prompt = "Summarize."

    elif intent == 'author':
        system_prompt = f"Context: {context_text}\n\nFind the author name."
        user_prompt = "Who is the author?"

    else:  # rag
        system_prompt = f"Context: {context_text}\n\nAnswer based ONLY on context. Cite as [Source: file (Page N)]."
        user_prompt = query

    try:
        llm = get_llm()
        for token in llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,  # Allow for longer responses
            temperature=0.7
        ):
            cleaned = token.replace("$", " USD ")
            yield cleaned
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        yield "An error occurred while generating the response."
