---
title: YouTube RAG Intelligence Bot
emoji: 📺
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.10
---

# 📺 YouTube Intelligence Bot (RAG-Powered)

This project utilises Retrieval-Augmented Generation (RAG) to summarize and query YouTube video content.

## 🏗 System Architecture
The application implements a **Modular RAG** architecture:
* **LLM**: Gemini 2.5 Flash
* **Embeddings**: gemini-embedding-001
* **Vector Store**: FAISS