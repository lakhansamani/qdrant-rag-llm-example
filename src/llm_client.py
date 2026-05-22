"""
llm_client.py
-------------
Thin wrapper around Ollama's HTTP API for local LLM inference.

Ollama runs open-source models (Llama 3, Mistral, Phi-3, Gemma 2, etc.)
entirely on your machine. No data ever leaves your network.

Installation:
    macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
    Windows:     Download from https://ollama.com/download

Pull a model before running:
    ollama pull llama3.2    # ~2GB, great for Q&A
    ollama pull mistral     # ~4GB, good reasoning
    ollama pull phi3        # ~2GB, fast & small

The Ollama server starts automatically when you run any `ollama` command.
Default endpoint: http://localhost:11434
"""

import json
from typing import Iterator
import urllib.request
import urllib.error


# The default Ollama server address. Override via environment variable in production.
OLLAMA_BASE_URL = "http://localhost:11434"

# The prompt template for RAG. The LLM is instructed to:
#   1. Answer only from the provided context.
#   2. Say "I don't know" if the context doesn't contain the answer.
#   3. Cite the source document.
RAG_SYSTEM_PROMPT = """You are a helpful internal knowledge base assistant.

Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say:
"I don't have enough information in the knowledge base to answer this question."

Always cite the source document(s) you used in your answer.
Be concise and accurate. Do not make up information."""


class OllamaClient:
    """
    Communicates with a locally running Ollama server.

    Uses only Python's built-in urllib — no extra dependencies required.

    Raises:
        ConnectionError: If Ollama is not running.
        RuntimeError:    If the model is not available (run `ollama pull <model>`).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.1,
        context_length: int = 4096,
    ) -> None:
        """
        Args:
            model:          Name of the Ollama model to use. Run `ollama list` to see
                            available models. Popular choices: llama3.2, mistral, phi3.
            base_url:       URL of the Ollama server. Default is localhost:11434.
            temperature:    Sampling temperature (0 = deterministic, 1 = creative).
                            Keep low (0.1) for factual Q&A tasks.
            context_length: Maximum context window in tokens.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.context_length = context_length

    def _post(self, endpoint: str, payload: dict) -> dict:
        """Send a JSON POST request to the Ollama API."""
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                available = ", ".join(self.list_models()) or "(none — run `ollama pull`)"
                raise RuntimeError(
                    f"Model '{self.model}' is not installed in Ollama. "
                    f"Available models: {available}. "
                    f"Run: ollama pull {self.model}"
                ) from e
            raise RuntimeError(
                f"Ollama API error {e.code} at {url}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve` or just `ollama run <model>`"
            ) from e

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            url = f"{self.base_url}/api/tags"
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return a list of model names available in Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def _model_installed(self, model: str, available: list[str]) -> bool:
        """True if `model` matches an Ollama tag (with or without :tag suffix)."""
        base = model.split(":")[0]
        for name in available:
            if name == model or name.split(":")[0] == base:
                return True
        return False

    def ensure_model_available(self) -> None:
        """
        Verify Ollama is reachable and the configured model is installed.

        Raises:
            ConnectionError: If the Ollama server is not running.
            RuntimeError:    If the model is not installed locally.
        """
        if not self.is_available():
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` or run any `ollama` command."
            )
        available = self.list_models()
        if not self._model_installed(self.model, available):
            models = ", ".join(available) or "(none — run `ollama pull <model>`)"
            raise RuntimeError(
                f"Model '{self.model}' is not installed in Ollama. "
                f"Available models: {models}. "
                f"Run: ollama pull {self.model}"
            )

    def generate(
        self,
        prompt: str,
        system: str = RAG_SYSTEM_PROMPT,
        stream: bool = False,
    ) -> str:
        """
        Generate a completion from the local LLM.

        Args:
            prompt: The user prompt (includes the retrieved context + question).
            system: The system prompt that sets the LLM's role and behaviour.
            stream: If True, streams output token by token (not yet implemented in UI).

        Returns:
            The model's response as a string.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,           # We wait for the full response
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_length,
            },
        }
        response = self._post("/api/generate", payload)
        return response.get("response", "").strip()

    def build_rag_prompt(self, context: str, question: str) -> str:
        """
        Build the full prompt for a RAG query.

        Injects the retrieved document context and the user's question
        into a structured prompt template.

        Args:
            context:  The formatted document chunks from the retriever.
            question: The user's original question.

        Returns:
            A complete prompt string ready to send to the LLM.
        """
        return (
            f"CONTEXT FROM KNOWLEDGE BASE:\n"
            f"{'=' * 50}\n"
            f"{context}\n"
            f"{'=' * 50}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )
