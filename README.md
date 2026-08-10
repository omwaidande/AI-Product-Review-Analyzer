# AI Product Review Analyzer with Streamlit

A simple Retrieval-Augmented Generation (RAG) app: upload a `.txt` document, ask questions, and get answers generated *only* from that document's content — powered by Groq's LLM API and local sentence embeddings.

## How it works

1. **Load** — the uploaded document is read from disk.
2. **Chunk** — the text is split into paragraphs (or fixed-size overlapping chunks as a fallback).
3. **Embed** — each chunk is turned into a vector using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieve** — when you ask a question, it's embedded too, and compared against every chunk using cosine similarity. The top 2 most relevant chunks are pulled out.
5. **Generate** — those chunks are passed as context to an LLM (via Groq's OpenAI-compatible API), which answers strictly from that context.

## Project structure

```
.
├── ITR_product_UI.py             # Streamlit frontend
├── ITR_project.py      # RAG backend (chunking, embedding, retrieval, generation)
├── .env                # Your API key (not committed to git)
├── .gitignore
└── README.md
```

## Requirements

- Python 3.9+
- A [Groq](https://console.groq.com/) API key

## Setup

1. Clone or download this repo, and open the folder in VS Code (or your editor of choice).

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install streamlit openai python-dotenv sentence-transformers
   ```

4. Create a `.env` file in the project root with your Groq API key:

   ```
   OPENAI_API_KEY=your_groq_api_key_here
   ```

   (The variable is named `OPENAI_API_KEY` because the app uses the OpenAI-compatible client pointed at Groq's endpoint — it is not an actual OpenAI key.)

## Running the app

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

1. Upload a `.txt` file using the sidebar.
2. Wait for it to finish chunking and embedding.
3. Ask questions in the chat box — answers are generated only from the uploaded document.

To stop the app, press `Ctrl + C` in the terminal.

## Notes & limitations

- Only `.txt` files are currently supported (no PDF/DOCX ingestion yet).
- Embeddings are kept in memory only — reloading the app re-embeds the document from scratch.
- Retrieval always returns the top 2 most similar chunks; this isn't currently configurable in the UI.
- No persistent vector store — fine for small/demo documents, not optimized for large corpora.

## License

Add your preferred license here (e.g. MIT).
