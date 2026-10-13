"""Recursos externos que usan las herramientas de los agentes.

Se arman UNA vez (al arrancar el worker o la CLI) y se inyectan al grafo. Las
herramientas los reciben por parametro, no los crean ni los buscan en
variables globales. Ventaja: en los tests se inyectan versiones simuladas
(Chroma de prueba, embeddings falsos, internet de juguete) sin tocar el
codigo de los agentes.
"""

from __future__ import annotations

from dataclasses import dataclass

from comun.config import Configuracion
from comun.web import BuscadorWeb, WebTavily
from rag.almacen import AlmacenChroma
from rag.catalogo import Catalogo
from rag.embeddings import construir_vectorizador
from rag.hibrido import BuscadorHibrido
from rag.indexar import asegurar_indice


@dataclass(frozen=True)
class Recursos:
    buscador: BuscadorHibrido   # RAG: catalogo y metodologia
    catalogo: Catalogo          # fichas completas, para validar servicios y tarifas
    web: BuscadorWeb            # Tavily


async def abrir_recursos(config: Configuracion, web: BuscadorWeb | None = None) -> Recursos:
    """Conecta Chroma, asegura que el indice este al dia y arma los recursos."""
    almacen = await AlmacenChroma.conectar(config, intentos=15)
    vectorizador = construir_vectorizador(config)
    informe = await asegurar_indice(config, almacen, vectorizador)
    for nombre, estado in informe.items():
        print(f"  RAG {nombre}: {estado}", flush=True)
    return Recursos(
        buscador=BuscadorHibrido(almacen, vectorizador, config.rag_top_k),
        catalogo=Catalogo(almacen),
        web=web or WebTavily(config),
    )
