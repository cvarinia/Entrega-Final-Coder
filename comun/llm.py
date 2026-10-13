"""Construccion del modelo de lenguaje (reutilizada de las pre-entregas 5 a 7).

Este es el UNICO archivo que sabe que proveedor de LLM se esta usando. El
Supervisor y los agentes reciben un `BaseChatModel` ya armado y no saben si
detras hay Claude o Gemini.

    PROVEEDOR=anthropic   -> Claude (el que se usa en la entrega final)
    PROVEEDOR=gemini      -> Gemini (alternativa sin costo)

Ojo: esto es solo el modelo que RAZONA. Los embeddings del RAG son siempre de
Gemini y viven en rag/embeddings.py: son dos roles independientes.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from comun.config import Configuracion

# 2048 se quedaba corto: el investigador escribe el perfil completo como
# argumento de entregar_perfil y la respuesta se cortaba (llegaba vacia: {}).
MAX_TOKENS_RESPUESTA = 4096


def construir_llm(config: Configuracion) -> BaseChatModel:
    """Instancia el modelo de chat del proveedor configurado.

    Las importaciones son perezosas para que el proyecto arranque aunque este
    instalado solo uno de los dos clientes.
    """
    if config.proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.modelo,
            temperature=config.temperatura,
            api_key=config.api_key,
            timeout=config.timeout_modelo,
            max_tokens=MAX_TOKENS_RESPUESTA,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.modelo,
        temperature=config.temperatura,
        google_api_key=config.api_key,
    )
