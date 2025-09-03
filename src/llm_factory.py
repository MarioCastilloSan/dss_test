# src/llm_factory.py
import os
import logging
from langchain_community.llms import LlamaCpp
from groq import Groq
from src.config import MODEL_PATH, MODEL_CONFIG

logger = logging.getLogger(__name__)

def get_llm(provider: str = "local"):
    """
    Factory to return an LLM depending on the provider.
    provider="local" -> LlamaCpp (Mistral .gguf)
    provider="groq"  -> Groq API (LLaMA-4 Scout 17B)
    """
    if provider == "local":
        logger.info("Using local LlamaCpp model")
        return LlamaCpp(model_path=MODEL_PATH, **MODEL_CONFIG)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")

        logger.info("Using Groq API (meta-llama/llama-4-scout-17b-16e-instruct)")
        return Groq(api_key=api_key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
