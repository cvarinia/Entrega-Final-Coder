"""Probar el RAG a mano, sin agentes (util para entender que recupera).

Uso:
    python -m rag.consultar "LSP que busca proveedores de Storyline en arabe"
    python -m rag.consultar "cuanto cobran el QA" --complementos
    python -m rag.consultar "como escribir el primer email" --coleccion metodologia --agente estratega_redactor

Muestra, para cada fragmento, en que puesto lo puso cada busqueda (vectorial
y BM25) y el puntaje combinado (RRF). Sirve para ver la busqueda hibrida en accion.
"""

from __future__ import annotations

import argparse
import asyncio

from comun.config import Configuracion
from rag.almacen import AlmacenChroma
from rag.embeddings import construir_vectorizador
from rag.hibrido import BuscadorHibrido, filtro_metodologia_para, filtro_servicios_ofrecibles


async def main(args: argparse.Namespace) -> None:
    config = Configuracion.desde_entorno()
    almacen = await AlmacenChroma.conectar(config)
    buscador = BuscadorHibrido(almacen, construir_vectorizador(config), config.rag_top_k)

    if args.coleccion == "catalogo":
        filtro = filtro_servicios_ofrecibles(incluir_complementos=args.complementos)
    else:
        filtro = filtro_metodologia_para(args.agente) if args.agente else None

    print(f"Consulta: {args.consulta}\nColeccion: {args.coleccion} | Filtro: {filtro}\n")
    for n, r in enumerate(await buscador.buscar(args.coleccion, args.consulta, filtro), start=1):
        print(f"{n}. {r.id}  RRF={r.puntaje_rrf:.4f}  "
              f"(vectorial: #{r.puesto_vectorial or '-'} | BM25: #{r.puesto_bm25 or '-'})")
        print("   " + r.texto.splitlines()[0])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consulta de prueba al RAG.")
    parser.add_argument("consulta")
    parser.add_argument("--coleccion", choices=["catalogo", "metodologia"], default="catalogo")
    parser.add_argument("--complementos", action="store_true", help="Incluir servicios 'complemento' (QA).")
    parser.add_argument("--agente", help="Filtrar metodologia por agente (ej. estratega_redactor).")
    asyncio.run(main(parser.parse_args()))
