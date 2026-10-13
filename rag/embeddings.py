"""Embeddings: convierten texto en vectores para la busqueda semantica.

Por que Gemini si el sistema razona con Claude:
    Anthropic no ofrece un modelo de embeddings. Son dos roles independientes:
      - los embeddings (Gemini) solo sirven para ENCONTRAR los fragmentos;
      - el LLM (Claude) LEE el texto de esos fragmentos y razona.
    Claude nunca ve un vector. La unica regla: indexar y buscar con el MISMO
    modelo de embeddings (cada modelo arma su propio "mapa" de significados).

Dos tipos de tarea (task_type) de Gemini:
    RETRIEVAL_DOCUMENT -> al indexar el corpus
    RETRIEVAL_QUERY    -> al vectorizar la pregunta
El modelo esta entrenado para que una pregunta quede cerca de los documentos
que la responden, aunque no compartan las mismas palabras.

Todo es asincrono: GoogleGenerativeAIEmbeddings usa el cliente async oficial
(client.aio), asi que vectorizar no bloquea el event loop.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from comun.config import Configuracion
from comun.reintentos import con_reintentos


class Vectorizador(Protocol):
    """Lo unico que el RAG le pide a un modelo de embeddings."""

    async def vectorizar_documentos(self, textos: list[str]) -> list[list[float]]: ...

    async def vectorizar_consulta(self, texto: str) -> list[float]: ...


class EmbeddingsGemini:
    """Embeddings de Gemini (gemini-embedding-001 por defecto, ver .env)."""

    def __init__(self, config: Configuracion) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._modelo = GoogleGenerativeAIEmbeddings(
            model=config.modelo_embeddings,
            google_api_key=config.google_api_key,
        )

    async def vectorizar_documentos(self, textos: list[str]) -> list[list[float]]:
        return await con_reintentos(
            lambda: self._modelo.aembed_documents(textos, task_type="RETRIEVAL_DOCUMENT")
        )

    async def vectorizar_consulta(self, texto: str) -> list[float]:
        return await con_reintentos(
            lambda: self._modelo.aembed_query(texto, task_type="RETRIEVAL_QUERY")
        )


class EmbeddingsSimulados:
    """Embeddings falsos y deterministas, para los tests (sin API ni costo).

    Cada palabra suma 1 en una posicion del vector elegida por su hash
    ("bolsa de palabras"). No entiende sinonimos, pero dos textos que comparten
    palabras quedan cerca: alcanza para probar el cableado del RAG.
    """

    DIMENSIONES = 256

    def _vector(self, texto: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONES
        for palabra in re.findall(r"\w+", texto.lower()):
            posicion = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % self.DIMENSIONES
            vector[posicion] += 1.0
        norma = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norma for v in vector]

    async def vectorizar_documentos(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]

    async def vectorizar_consulta(self, texto: str) -> list[float]:
        return self._vector(texto)


def construir_vectorizador(config: Configuracion) -> Vectorizador:
    """Fabrica: simulado en los tests, Gemini en el resto."""
    if config.modelo_embeddings == "embeddings-simulados":
        return EmbeddingsSimulados()
    return EmbeddingsGemini(config)
