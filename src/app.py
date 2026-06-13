"""
app.py
------
Gradio web UI for the Local RAG Knowledge Base.

Starts a local web server (default: http://localhost:7860) with:
  - A chat interface for asking questions
  - A sidebar showing retrieved document sources

With --authorizer, the app is permission-aware: a LOGIN SCREEN is shown first
and the chat stays hidden until the user authenticates. Each question then runs
with the logged-in user's token, so retrieval is restricted to the documents
that user can_view (see src/authz.py).

Run with:
    python src/app.py
    python src/app.py --model mistral --storage ./qdrant_data
    python src/app.py --authorizer http://localhost:8080   # permission-aware
"""

import argparse
import sys
from pathlib import Path

# Project root on sys.path so `python src/app.py` works from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.authz import AuthorizationError, AuthzClient
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


def build_app(
    pipeline: RAGPipeline,
    storage: str = DEFAULT_STORAGE,
    authz: AuthzClient | None = None,
    client_id: str = "",
) -> gr.Blocks:
    """
    Build and return the Gradio application.

    Without --authorizer the chat is shown immediately (single-user mode).
    With --authorizer the app gates the chat behind a login screen:
      - the login screen is shown first; the chat column is hidden;
      - the user logs in via Authorizer's hosted page (a redirect) OR with
        email/password against the Authorizer API;
      - on success the login screen is hidden and the chat is revealed;
      - every question runs with that user's token (permission-filtered).
    """
    fga = authz is not None
    base_url = authz.base_url if fga else ""
    dashboard_url = _qdrant_dashboard_url(storage)

    with gr.Blocks(title="🔍 Local RAG Knowledge Base") as demo:

        # ── Header ────────────────────────────────────────────────────────
        header = (
            "# 🔍 Local RAG Knowledge Base\n"
            "**Ask questions about your company's internal documents.** "
            "Everything runs locally — your data never leaves your machine.\n\n"
            "> 💡 *Qdrant (vector search) · FastEmbed (embeddings) · Ollama (LLM)*"
        )
        if fga:
            header += " *· Authorizer + OpenFGA (permissions)*"
        if dashboard_url:
            header += f"\n>\n> 🧭 *Qdrant dashboard:* [{dashboard_url}]({dashboard_url})"
        gr.Markdown(header)

        token_state = gr.State("")

        # ── LOGIN SCREEN (FGA mode only; shown first, chat hidden) ─────────
        login_view = None
        login_status = None
        if fga:
            with gr.Column(visible=True) as login_view:
                gr.Markdown(
                    "## 🔐 Log in to continue\n"
                    "This knowledge base is **permission-aware** — you only get answers "
                    "from documents you're allowed to view. Log in, then the chat appears.\n\n"
                    "> **Demo accounts** (password `Demo@Pass123`):\n"
                    "> - `alice@example.com` — engineering → onboarding guide + tech stack\n"
                    "> - `carol@example.com` — finance → onboarding guide + financial report\n"
                    "> - `bob@example.com` — new hire → onboarding guide only\n"
                    ">\n"
                    "> Then ask *“What was our Q4 revenue?”* — Alice (engineering) is blocked "
                    "from the finance report; Carol (finance) gets the answer."
                )
                login_status = gr.Textbox(
                    label="Status", value="Checking your session…",
                    interactive=False,
                )
                hosted_btn = gr.Button(
                    "🔐 Log in with Authorizer (hosted login page)", variant="primary"
                )
                with gr.Accordion("…or log in with email & password", open=True):
                    email_box = gr.Textbox(label="Email", placeholder="alice@example.com")
                    password_box = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Log in", variant="secondary")

        # ── CHAT SCREEN (visible immediately without auth; gated with auth) ─
        with gr.Column(visible=not fga) as chat_view:

            if fga:
                with gr.Row():
                    session_banner = gr.Markdown("✅ Logged in.")
                    logout_btn = gr.Button("Log out 🔒", scale=0)

            # ── Status bar ─────────────────────────────────────────────────
            with gr.Row():
                doc_count = gr.Textbox(
                    label="📚 Knowledge Base Status",
                    value=f"{pipeline.document_count} chunks indexed | Model: {pipeline.llm.model}",
                    interactive=False,
                    scale=3,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            if fga:
                gr.Markdown(
                    "### 💬 Ask a question\n"
                    "Ask *“What was our Q4 revenue?”*. As **Alice** (engineering) the finance "
                    "report is never retrieved, so the assistant says it has no information. "
                    "Log out and back in as **Carol** (finance) to get the numbers."
                )

            # ── Main chat area ─────────────────────────────────────────────
            with gr.Row():
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

                    gr.Examples(
                        examples=[
                            ["What was our Q4 revenue and cash runway?"],
                            ["What tech stack do we use for the frontend?"],
                            ["How do I report a security incident?"],
                            ["What is the annual leave entitlement?"],
                            ["How do I get access to GitHub?"],
                            ["What is our data residency policy?"],
                        ],
                        inputs=question_box,
                        label="Example questions",
                    )

                with gr.Column(scale=2):
                    sources_panel = gr.Markdown(
                        value="*Ask a question to see which documents were used.*",
                        label="📎 Retrieved Sources",
                    )

            with gr.Accordion("ℹ️ How this works", open=False):
                gr.Markdown(
                    "**Retrieval-Augmented Generation (RAG)**: your question is embedded "
                    "with FastEmbed, the most similar chunks are found in Qdrant, and a "
                    "local Ollama model answers from them. "
                    + (
                        "With permissions on, the search is restricted to documents you "
                        "may view — Authorizer's embedded OpenFGA returns your allow-list, "
                        "and it becomes a Qdrant payload filter, so off-limits documents "
                        "are never even retrieved."
                        if fga else
                        "No data leaves your machine."
                    )
                )

        # ── Question handler ──────────────────────────────────────────────
        def ask_question(question: str, history: list | None, token: str):
            """Run the RAG pipeline for one question; update chat + sources."""
            messages = list(history or [])
            if not question.strip():
                return messages, "*Please enter a question.*", ""
            if fga and not token:
                return messages, "*🔐 Please log in first.*", question

            try:
                response = pipeline.ask(question, user_token=token or None)
            except (RuntimeError, ConnectionError, AuthorizationError) as e:
                messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"❌ {e}"},
                ])
                return messages, f"*Error: {e}*", ""

            sources_md = "### 📎 Sources Used\n\n"
            if response.sources:
                for i, chunk in enumerate(response.sources, 1):
                    snippet = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
                    sources_md += (
                        f"**{i}. {chunk.source}** *(similarity: {chunk.score:.2%})*\n\n"
                        f"> {snippet}\n\n---\n\n"
                    )
            else:
                sources_md += "*No documents you can access were relevant to this query.*"

            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.answer},
            ])
            return messages, sources_md, ""

        def refresh_status():
            return f"{pipeline.document_count} chunks indexed | Model: {pipeline.llm.model}"

        submit_btn.click(
            ask_question, [question_box, chatbot, token_state],
            [chatbot, sources_panel, question_box],
        )
        question_box.submit(
            ask_question, [question_box, chatbot, token_state],
            [chatbot, sources_panel, question_box],
        )
        refresh_btn.click(refresh_status, outputs=doc_count)

        # ── Auth wiring (FGA mode only) ───────────────────────────────────
        if fga:
            def reveal(token: str):
                """Show the chat and hide the login screen once a token exists."""
                logged_in = bool(token)
                return (
                    gr.update(visible=not logged_in),   # login_view
                    gr.update(visible=logged_in),        # chat_view
                )

            # On page load: silently check for an existing session, or exchange
            # the ?code returned from the hosted login page. No auto-redirect —
            # the login screen (with the demo-account hints) is shown first.
            load_js = (
                "async () => {"
                "  if (typeof authorizerdev === 'undefined')"
                "    return ['', '⚠️ Could not load the Authorizer script (offline?). "
                "Use email & password below.'];"
                "  try {"
                f"    const ref = new authorizerdev.Authorizer({{ authorizerURL: '{base_url}',"
                "       redirectURL: window.location.origin + window.location.pathname,"
                f"      clientID: '{client_id}' }});"
                "    let token = '';"
                "    if (window.location.search.indexOf('code=') !== -1) {"
                "      const r = await ref.authorize({ response_type: 'code', use_refresh_token: false });"
                "      token = ((r && r.data) || r || {}).access_token || '';"
                "      history.replaceState({}, '', window.location.pathname);"
                "    } else {"
                "      const s = await ref.getSession();"
                "      token = ((s && s.data) || s || {}).access_token || '';"
                "    }"
                "    if (token) {"
                "      const p = await ref.getProfile({ Authorization: 'Bearer ' + token });"
                "      const email = ((p && p.data) || p || {}).email || 'user';"
                "      return [token, 'Logged in as ' + email];"
                "    }"
                "  } catch (e) { console.error('authorizer session check failed', e); }"
                "  return ['', 'Please log in to continue.'];"
                "}"
            )
            demo.load(
                None, None, [token_state, login_status], js=load_js,
            ).then(reveal, token_state, [login_view, chat_view])

            # Hosted login: redirect the browser to Authorizer's login page.
            # Runs on a user click (a real gesture), so the redirect is reliable.
            hosted_js = (
                "async () => {"
                "  if (typeof authorizerdev === 'undefined') {"
                "    console.error('Authorizer script not loaded'); return; }"
                f"  const ref = new authorizerdev.Authorizer({{ authorizerURL: '{base_url}',"
                "     redirectURL: window.location.origin + window.location.pathname,"
                f"    clientID: '{client_id}' }});"
                "  await ref.authorize({ response_type: 'code', use_refresh_token: false });"
                "}"
            )
            hosted_btn.click(None, None, None, js=hosted_js)

            # Email/password login against the Authorizer API (always works,
            # no redirect — the reliable path and the offline fallback).
            def do_login(email: str, password: str):
                try:
                    token = authz.login(email.strip(), password)
                    allowed = authz.allowed_documents(token)
                    docs = ", ".join(sorted(allowed)) if allowed else "no documents"
                    return token, f"Logged in as {email.strip()} — may view: {docs}"
                except AuthorizationError as e:
                    return "", f"Login failed: {e}"

            login_btn.click(
                do_login, [email_box, password_box], [token_state, login_status],
            ).then(reveal, token_state, [login_view, chat_view])

            def greet(token: str):
                return "✅ Logged in." if token else ""

            token_state.change(greet, token_state, session_banner)

            # Log out: clear the Authorizer session and return to the login screen.
            logout_js = (
                "async () => {"
                "  try {"
                f"    const ref = new authorizerdev.Authorizer({{ authorizerURL: '{base_url}',"
                "       redirectURL: window.location.origin + window.location.pathname,"
                f"      clientID: '{client_id}' }});"
                "    await ref.logout();"
                "  } catch (e) {}"
                "  window.location.href = window.location.pathname;"
                "}"
            )
            logout_btn.click(None, None, None, js=logout_js)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Local RAG Knowledge Base UI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Qdrant storage path")
    parser.add_argument("--data", default=str(DATA_DIR), help="Knowledge base directory")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument(
        "--authorizer",
        default=None,
        help="Authorizer server URL (e.g. http://localhost:8080). Enables "
             "fine-grained permissions: login required, retrieval restricted "
             "to documents the user can_view. Seed first: python scripts/fga_seed.py",
    )
    parser.add_argument(
        "--client-id",
        default="123456",
        help="Authorizer client id for the hosted login flow "
             "(value of the server's --client-id flag; see docker-compose.yml)",
    )
    args = parser.parse_args()

    # Build the pipeline and ingest documents
    authz = AuthzClient(args.authorizer) if args.authorizer else None
    pipeline = RAGPipeline(
        llm_model=args.model,
        storage_path=args.storage,
        authz=authz,
    )
    try:
        pipeline.llm.ensure_model_available()
    except (RuntimeError, ConnectionError) as e:
        raise SystemExit(f"\n{e}\n") from e
    pipeline.ingest_directory(Path(args.data))

    # Build and launch the Gradio UI
    app = build_app(
        pipeline, storage=args.storage, authz=authz, client_id=args.client_id
    )
    # In FGA mode, load authorizer-js from the CDN so the browser can run the
    # hosted-login flow. Passed to launch() (Gradio 6 moved `head` here).
    head_html = (
        '<script src="https://unpkg.com/@authorizerdev/authorizer-js/'
        'lib/authorizer.min.js"></script>'
        if authz
        else None
    )
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,       # Auto-open the browser
        head=head_html,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()
