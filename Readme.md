---
title: YouTube RAG Intelligence Bot
emoji: 📺
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "5.0.1"
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.10"
---
# YouTube Intelligence Bot (RAG-Powered)

This project utilises Retrieval-Augmented Generation (RAG) to summarize and query YouTube video content.

# LIVE DEMO
https://yt-rag-bot.onrender.com

## System Architecture
The application implements a **Modular RAG** architecture:
* **LLM**: Gemini 2.5 Flash
* **Embeddings**: gemini-embedding-001
* **Vector Store**: FAISS

```mermaid
flowchart TD
    A([User query + YouTube URL]) --> B

    B["**Ingestion**\nYouTubeTranscriptApi → raw transcript text"]
    B --> C

    C["**Processing**\nRecursiveCharacterTextSplitter → overlapping chunks"]
    C --> D

    D["**Embedding**\ngemini-embedding-001 → high-dimensional vectors"]
    D --> E

    E[("**Storage**\nFAISS vector DB — stores embeddings in local memory")]
    E --> F

    F["**Retrieval**\nCosine similarity search → top-k relevant chunks"]
    F --> G

    G["**Generation**\nGemini 2.5 Flash synthesizes final answer\nwith retrieved context"]
    G --> H([Answer returned to user])

    style B fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style C fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style D fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style E fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style F fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style G fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
```

> **Note**: Due to YouTube's IP restrictions on cloud providers, 
> the hosted demo may not fetch transcripts. Clone and run locally 
> for full functionality. The RAG architecture is fully implemented.
