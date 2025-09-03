# src/rag_agent.py
import json
import logging
from typing import Optional, Dict, Any

from langchain_community.llms import LlamaCpp
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.config import MODEL_PATH, MODEL_CONFIG, TARGET_QUESTION, DOCS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAGAgent:
    """
    Retrieval-Augmented Generation agent:
    retrieve relevant docs from Qdrant -> build prompt -> query LLM -> structured answer.
    """

    def __init__(self):
        dp = DocumentProcessor(DOCS_DIR)  # only used for embeddings model
        self.vector_store = VectorStore(dp.embeddings)
        self.llm = LlamaCpp(model_path=MODEL_PATH, **MODEL_CONFIG)

    def _build_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful assistant. Always answer in Spanish.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1) Answer concisely in Spanish.\n"
            "2) Return ONLY a JSON object with keys: "
            '"respuesta", "documento_referencia", "pagina_referencia".\n'
            "3) If information is missing, use 'N/A'.\n\n"
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

        # Build context string
        context = "\n\n".join(
            f"[{i+1}] {d['metadata'].get('source', 'N/A')} (page {d['metadata'].get('page', 'N/A')}):\n{d['page_content']}"
            for i, d in enumerate(docs)
        )

        prompt = self._build_prompt(context, question)

        try:
            raw_output = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
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
        """
        Shortcut to run the pre-defined question from config.
        """
        return self.query(TARGET_QUESTION)
