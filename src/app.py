"""
app.py
------
Gradio web UI for the Local RAG Knowledge Base.

Starts a local web server (default: http://localhost:7860) with:
  - A chat interface for asking questions
  - A sidebar showing retrieved document sources
  - An ingestion panel to add new documents at runtime

Run with:
    python src/app.py
    python src/app.py --model mistral --storage ./qdrant_data
"""

import argparse
import sys
from pathlib import Path

# Project root on sys.path so `python src/app.py` works from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.pipeline import RAGPipeline

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
DEFAULT_MODEL = "llama3.2"
# Default to the Qdrant Docker container so users get the built-in dashboard UI
# at http://localhost:6333/dashboard. Override with ":memory:" or "./qdrant_data".
DEFAULT_STORAGE = "http://localhost:6333"


def _qdrant_dashboard_url(storage: str) -> str | None:
    """Return the dashboard URL when connected to a Qdrant HTTP server, else None."""
    if storage.startswith(("http://", "https://")):
        return storage.rstrip("/") + "/dashboard"
    return None


def build_app(pipeline: RAGPipeline, storage: str = DEFAULT_STORAGE) -> gr.Blocks:
    """
    Build and return the Gradio application.

    The app has three sections:
      1. Chat — ask questions, see answers and sources
      2. Status — see how many documents are indexed
      3. About — brief explanation of what the system does
    """
    dashboard_url = _qdrant_dashboard_url(storage)

    with gr.Blocks(title="🔍 Local RAG Knowledge Base") as demo:

        # ── Header ────────────────────────────────────────────────────────
        header = """
        # 🔍 Local RAG Knowledge Base
        **Ask questions about your company's internal documents.**
        Everything runs locally — your data never leaves your machine.

        > 💡 *Powered by Qdrant (vector search) + FastEmbed (embeddings) + Ollama (LLM)*
        """
        if dashboard_url:
            header += (
                f"\n> 🧭 *Inspect the embeddings in the Qdrant dashboard:* "
                f"[{dashboard_url}]({dashboard_url})\n"
            )
        gr.Markdown(header)

        # ── Status bar ───────────────────────────────────────────────────
        with gr.Row():
            doc_count = gr.Textbox(
                label="📚 Knowledge Base Status",
                value=f"{pipeline.document_count} chunks indexed | Model: {pipeline.llm.model}",
                interactive=False,
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        # ── Main chat area ────────────────────────────────────────────────
        with gr.Row():
            # Left: Chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=450)
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Ask anything about the knowledge base...",
                        label="Your question",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("Ask ➤", variant="primary", scale=1)

                # Example questions to help users get started
                gr.Examples(
                    examples=[
                        ["How do I report a security incident?"],
                        ["What is our data residency policy?"],
                        ["What is the annual leave entitlement?"],
                        ["What tech stack do we use for the frontend?"],
                        ["Are we allowed to send customer data to AI APIs?"],
                        ["How do I get access to GitHub?"],
                    ],
                    inputs=question_box,
                    label="Example questions",
                )

            # Right: Source panel
            with gr.Column(scale=2):
                sources_panel = gr.Markdown(
                    value="*Ask a question to see which documents were used.*",
                    label="📎 Retrieved Sources",
                )

        # ── Event handlers ────────────────────────────────────────────────

        def ask_question(question: str, history: list | None):
            """Handle a user question: run RAG pipeline, update chat + sources."""
            messages = list(history or [])
            if not question.strip():
                return messages, "*Please enter a question.*", ""

            # Run the full RAG pipeline
            try:
                response = pipeline.ask(question)
            except (RuntimeError, ConnectionError) as e:
                error_msg = f"❌ {e}"
                messages.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": error_msg},
                    ]
                )
                return messages, f"*Error: {e}*", ""

            # Format the sources panel
            sources_md = "### 📎 Sources Used\n\n"
            if response.sources:
                for i, chunk in enumerate(response.sources, 1):
                    sources_md += (
                        f"**{i}. {chunk.source}** *(similarity: {chunk.score:.2%})*\n\n"
                        f"> {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}\n\n"
                        f"---\n\n"
                    )
            else:
                sources_md += "*No relevant documents found for this query.*"

            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response.answer},
                ]
            )
            return messages, sources_md, ""

        def refresh_status():
            return f"{pipeline.document_count} chunks indexed | Model: {pipeline.llm.model}"

        # Wire up events
        submit_btn.click(
            fn=ask_question,
            inputs=[question_box, chatbot],
            outputs=[chatbot, sources_panel, question_box],
        )
        question_box.submit(
            fn=ask_question,
            inputs=[question_box, chatbot],
            outputs=[chatbot, sources_panel, question_box],
        )
        refresh_btn.click(fn=refresh_status, outputs=doc_count)

        # ── About section ─────────────────────────────────────────────────
        with gr.Accordion("ℹ️ How this works", open=False):
            gr.Markdown("""
            ### RAG Architecture

            This demo implements **Retrieval-Augmented Generation (RAG)**:

            1. **Ingestion** (done at startup):
               - Documents are split into overlapping text chunks (~400 chars)
               - Each chunk is embedded using **FastEmbed** (BAAI/bge-small-en-v1.5)
               - Vectors are stored in **Qdrant** (local, in-memory)

            2. **Retrieval** (on each question):
               - Your question is embedded using the same model
               - Qdrant finds the top-4 most similar document chunks (cosine similarity)
               - Chunks with similarity < 0.3 are filtered out

            3. **Generation** (on each question):
               - Retrieved chunks are formatted as context
               - The full prompt (context + question) is sent to **Ollama** (local LLM)
               - The LLM generates an answer grounded only in the provided context

            **Privacy**: No data leaves your machine. Qdrant, FastEmbed, and Ollama all run locally.
            """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Local RAG Knowledge Base UI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Qdrant storage path")
    parser.add_argument("--data", default=str(DATA_DIR), help="Knowledge base directory")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    # Build the pipeline and ingest documents
    pipeline = RAGPipeline(
        llm_model=args.model,
        storage_path=args.storage,
    )
    try:
        pipeline.llm.ensure_model_available()
    except (RuntimeError, ConnectionError) as e:
        raise SystemExit(f"\n{e}\n") from e
    pipeline.ingest_directory(Path(args.data))

    # Build and launch the Gradio UI
    app = build_app(pipeline, storage=args.storage)
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,       # Auto-open the browser
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()
