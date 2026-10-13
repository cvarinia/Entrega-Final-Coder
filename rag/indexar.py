"""Indexado del corpus en Chroma.

Uso:
    python -m rag.indexar            # indexa solo si el corpus cambio
    python -m rag.indexar --forzar   # re-indexa siempre

El worker llama a `asegurar_indice()` al arrancar, asi que con Docker Compose
no hace falta correr esto a mano: el primer arranque indexa y los siguientes
detectan que nada cambio (misma huella) y no gastan llamadas a la API.
"""

from __future__ import annotations

import argparse
import asyncio

from comun.config import Configuracion
from rag.almacen import AlmacenChroma
from rag.corpus import COLECCIONES, cargar_corpus
from rag.embeddings import Vectorizador, construir_vectorizador


async def asegurar_indice(
    config: Configuracion,
    almacen: AlmacenChroma,
    vectorizador: Vectorizador,
    forzar: bool = False,
) -> dict[str, str]:
    """Indexa las colecciones que lo necesiten. Devuelve que paso con cada una."""
    corpus = await cargar_corpus(config.ruta_corpus)
    informe: dict[str, str] = {}
    for coleccion in COLECCIONES:
        fragmentos = corpus[coleccion]
        reindexo = await almacen.indexar(
            coleccion, fragmentos, vectorizador, config.modelo_embeddings, forzar=forzar
        )
        estado = "indexada" if reindexo else "sin cambios"
        informe[almacen.nombre(coleccion)] = f"{estado} ({len(fragmentos)} fragmentos)"
    return informe


async def main(forzar: bool) -> None:
    config = Configuracion.desde_entorno()
    almacen = await AlmacenChroma.conectar(config)
    vectorizador = construir_vectorizador(config)
    print(f"Embeddings: {config.modelo_embeddings} | Chroma: {config.chroma_host}:{config.chroma_port}")
    informe = await asegurar_indice(config, almacen, vectorizador, forzar)
    for nombre, estado in informe.items():
        print(f"  {nombre}: {estado}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa el corpus de Headway en Chroma.")
    parser.add_argument("--forzar", action="store_true", help="Re-indexa aunque nada haya cambiado.")
    asyncio.run(main(parser.parse_args().forzar))
