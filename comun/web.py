"""Busqueda web del Agente Investigador (Tavily).

Tavily es un buscador pensado para agentes: en lugar de devolver una pagina de
resultados para humanos, devuelve JSON con titulo, URL y un extracto limpio.
Se usa su cliente ASINCRONO oficial (AsyncTavilyClient): nada de `requests`
adentro de una funcion async, el error que advierte la consigna.

Igual que con los embeddings, hay una version simulada para los tests: el
agente no sabe si del otro lado esta Tavily o un diccionario en memoria.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from comun.config import Configuracion
from comun.reintentos import con_reintentos

# Cuanto texto de una pagina se le pasa al agente. Mas texto = mas tokens y
# mas costo; con los primeros miles de caracteres alcanza para un perfil.
LIMITE_CARACTERES_PAGINA = 3000
LIMITE_CARACTERES_EXTRACTO = 600


def limpiar_pagina(texto: str) -> str:
    """Saca menus, botones y lineas sueltas: solo quedan las frases con contenido.

    En la primera prueba real, cada pagina empezaba con el menu del sitio
    ("Company / Services / AI Hub / ..."): tokens que el agente paga y no usa.
    """
    lineas = [l.strip() for l in texto.splitlines()]
    utiles = [l for l in lineas if len(l.split()) >= 6 or len(l) >= 60]
    return "\n".join(dict.fromkeys(utiles))  # dict.fromkeys: saca repetidas sin cambiar el orden


class ResultadoWeb(BaseModel):
    titulo: str
    url: str
    contenido: str


class BuscadorWeb(Protocol):
    async def buscar(self, consulta: str, max_resultados: int = 5) -> list[ResultadoWeb]: ...

    async def leer(self, url: str) -> str: ...


class WebTavily:
    def __init__(self, config: Configuracion) -> None:
        if not config.tavily_api_key or len(config.tavily_api_key) < 20:
            raise RuntimeError(
                "Falta TAVILY_API_KEY en el .env (busqueda web del Agente Investigador). "
                "Clave gratuita en https://app.tavily.com"
            )
        from tavily import AsyncTavilyClient

        self._cliente = AsyncTavilyClient(api_key=config.tavily_api_key)

    async def buscar(self, consulta: str, max_resultados: int = 5) -> list[ResultadoWeb]:
        respuesta = await con_reintentos(
            lambda: self._cliente.search(query=consulta, max_results=max_resultados, search_depth="basic")
        )
        return [
            ResultadoWeb(
                titulo=r.get("title", ""),
                url=r.get("url", ""),
                contenido=(r.get("content") or "")[:LIMITE_CARACTERES_EXTRACTO],
            )
            for r in respuesta.get("results", [])
        ]

    async def leer(self, url: str) -> str:
        respuesta = await con_reintentos(lambda: self._cliente.extract(urls=[url], format="text"))
        resultados = respuesta.get("results") or []
        if not resultados:
            raise ValueError(f"No se pudo leer la pagina {url}.")
        return limpiar_pagina(resultados[0].get("raw_content") or "")[:LIMITE_CARACTERES_PAGINA]


class WebSimulada:
    """Internet de juguete para los tests: paginas fijas en un diccionario."""

    def __init__(self, paginas: dict[str, tuple[str, str]]) -> None:
        # url -> (titulo, contenido)
        self.paginas = paginas
        self.consultas: list[str] = []

    async def buscar(self, consulta: str, max_resultados: int = 5) -> list[ResultadoWeb]:
        self.consultas.append(consulta)
        return [
            ResultadoWeb(titulo=t, url=u, contenido=c[:LIMITE_CARACTERES_EXTRACTO])
            for u, (t, c) in list(self.paginas.items())[:max_resultados]
        ]

    async def leer(self, url: str) -> str:
        if url not in self.paginas:
            raise ValueError(f"No se pudo leer la pagina {url}.")
        return self.paginas[url][1]
