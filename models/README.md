# Local AI Models

This directory stores the downloaded model weights.

## What is in here?

When you run the application for the first time, it downloads two models to this folder:

1.  **LLM**: `Qwen/Qwen2.5-0.5B-Instruct`
    -   Location: `models/llm/`
    -   Size: ~500MB
    -   Usage: Generates the answers in the chat.

2.  **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
    -   Location: `models/embeddings/`
    -   Size: ~100MB
    -   Usage: Turns your text into vectors for search.

## Git Configuration

We have configured `.gitignore` to ignore large model files in this directory. Do not try to commit them to GitHub as likely they exceed the file size limits.

If you delete this folder, the models will simply download again the next time you run `main.py`.
