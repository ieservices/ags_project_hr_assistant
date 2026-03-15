# Nestlé HR Assistant

A conversational chatbot that answers employee questions about Nestlé's HR policies using **Retrieval Augmented Generation (RAG)**.

## How it works

1. **Load** - The Nestlé HR Policy PDF is loaded and split into text chunks.
2. **Embed** - Each chunk is converted to a vector using OpenAI embeddings and stored in a FAISS index.
3. **Retrieve** - On each question, the most relevant chunks are retrieved by similarity search.
4. **Generate** - GPT-4o receives the retrieved context and produces a grounded, cited answer.
5. **Chat** - A Gradio web UI provides a conversational interface with memory across turns.

## Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS |
| RAG pipeline | LangChain (LCEL) |
| UI | Gradio |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
# Open .env and set: OPENAI_API_KEY=sk-...

# 3. Run
python hr_assistant.py
```

The app opens at `http://localhost:7860`.

## Project structure

```
project_hr_assistant/
├── hr_assistant.py   # Main application
├── requirements.txt
├── .env.example
└── dataset/
    └── the_nestle_hr_policy_pdf_2012.pdf
```
