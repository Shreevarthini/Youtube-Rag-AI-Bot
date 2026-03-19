---
title: Youtube RAG AI BOT
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: mit
---
# YouTube Intelligence Bot (RAG-Powered)

This project utilises Retrieval-Augmented Generation (RAG) to summarise and query YouTube video content.

## Live Demo


## 🛠 Tech Stack
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: gemini-embedding-001
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain (LCEL)
- **UI**: Gradio

## Example Output
**Video**: "Life at 40"
**Summary**: *The speaker reflects on turning 41, emphasizing that aging is "badass" and a time for increased confidence, contrasting it with outdated stereotypes...*

## CI/CD Pipeline
This project uses **GitHub Actions** for automated testing and deployment:
1. **Linting**: Checks code quality via `flake8`.
2. **Auto-Deploy**: On every `push` to `main`, the application is automatically rebuilt and deployed to the cloud.

## How to Run Locally
1. Clone the repo: `git clone <repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Set Environment Variable: `export GOOGLE_API_KEY='your_key_here'`
4. Run: `python ytbot.py`


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