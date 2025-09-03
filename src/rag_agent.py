# src/rag_agent.py
import re
import json
import logging
from typing import Optional, Dict, Any

from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.llm_factory import get_llm
from src.config import TARGET_QUESTION, DOCS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_json(text: str) -> dict:
    """Extract the first valid JSON object from a text string."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}")
    return {}


class RAGAgent:
    """
    Retrieval-Augmented Generation agent:
    - Retrieve relevant docs from Qdrant
    - Build prompt
    - Query LLM (local or Groq)
    - Return structured answer
    """

    def __init__(self, provider: str = "local"):
        dp = DocumentProcessor(DOCS_DIR)  # embeddings only
        self.vector_store = VectorStore(dp.embeddings)
        self.provider = provider
        self.llm = get_llm(provider)

    def _build_prompt(self, context: str, question: str) -> str:
        return (
            "Eres un asistente útil que responde SIEMPRE en español.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n\n"
            "Instrucciones:\n"
            "1) Devuelve una respuesta detallada en español integrando la información de TODOS los fragmentos relevantes.\n"
            "2) Resume claramente en qué proyectos fue relevante la chinchilla chinchilla.\n"
            "3) Devuelve SOLO un objeto JSON con las claves: "
            '"respuesta", "documento_referencia", "pagina_referencia".\n'
            "4) Si hay múltiples fragmentos, combínalos en una única 'respuesta'.\n"
            "5) Si falta información, usa 'N/A'.\n\n"
            "JSON:\n"
        )

    def query(self, question: str, region: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
        """
        Perform a RAG query:
        - Retrieve top-k documents (optionally filtered by region)
        - Ask the LLM with context
        - Return structured answer
        """
        docs = self.vector_store.similarity_search(question, k=k, region=region)
        if not docs:
            return {
                "respuesta": "No se encontraron documentos relevantes.",
                "documento_referencia": "N/A",
                "pagina_referencia": "N/A",
            }

        # Build context
        context = "\n\n".join(
            f"[{i+1}] {d['metadata'].get('source', 'N/A')} "
            f"(página {d['metadata'].get('page', 'N/A')}):\n{d['page_content']}"
            for i, d in enumerate(docs)
        )
        prompt = self._build_prompt(context, question)

        try:
            if self.provider == "groq":
                # Call Groq API
                response = self.llm.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil que responde en español. Devuelve SOLO JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                )
                raw_output = response.choices[0].message.content
            else:
                # Local LlamaCpp
                raw_output = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)

            # Try extracting JSON
            parsed = extract_json(raw_output)

            if not parsed:
                logger.warning("No valid JSON found in LLM response, returning raw text")
                return {
                    "respuesta": raw_output.strip(),
                    "documento_referencia": "N/A",
                    "pagina_referencia": "N/A",
                }

        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return {
                "respuesta": "Error generating response.",
                "documento_referencia": "N/A",
                "pagina_referencia": "N/A",
            }

        # Normalize JSON output
        return {
            "respuesta": parsed.get("respuesta", "N/A"),
            "documento_referencia": parsed.get("documento_referencia", "N/A"),
            "pagina_referencia": parsed.get("pagina_referencia", "N/A"),
        }

    def run_target_question(self):
        return self.query(TARGET_QUESTION)
