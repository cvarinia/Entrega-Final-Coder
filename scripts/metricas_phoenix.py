"""Metricas de un grupo de trazas en Phoenix (reutilizado del script de carga de la PE7).

Consulta la API GraphQL de Phoenix (la misma que usa su dashboard) filtrando
por la etiqueta del grupo de pruebas: costo, tokens y latencia p50/p95.
"""

from __future__ import annotations

import asyncio

import httpx

CONSULTA_PROYECTOS = "{ projects { edges { node { id name } } } }"
CONSULTA_METRICAS = """
query($id: ID!, $filtro: String) {
  node(id: $id) {
    ... on Project {
      costSummary(filterCondition: $filtro) {
        prompt { cost tokens } completion { cost tokens } total { cost tokens }
      }
      p50: latencyMsQuantile(probability: 0.5, filterCondition: $filtro)
      p95: latencyMsQuantile(probability: 0.95, filterCondition: $filtro)
    }
  }
}"""


async def metricas_phoenix(url: str, proyecto: str, etiqueta: str, esperadas: int) -> dict | None:
    """Consulta la API de Phoenix (la misma que usa su dashboard)."""
    filtro = f"metadata['etiqueta'] == '{etiqueta}'"
    async with httpx.AsyncClient(base_url=url, timeout=15) as phoenix:

        async def gql(consulta: str, variables: dict | None = None) -> dict:
            r = await phoenix.post("/graphql", json={"query": consulta, "variables": variables or {}})
            r.raise_for_status()
            return r.json()["data"]

        # El worker manda los spans en lotes (cada ~5 s): puede que las trazas
        # (y hasta el proyecto, la primera vez) tarden unos segundos en aparecer.
        trazas, datos = 0, None
        for _ in range(15):
            edges = (await gql(CONSULTA_PROYECTOS))["projects"]["edges"]
            proyectos = {e["node"]["name"]: e["node"]["id"] for e in edges}
            if proyecto in proyectos:
                trazas = await contar_trazas(phoenix, proyectos[proyecto], filtro)
                if trazas >= esperadas:
                    break
            await asyncio.sleep(2)

        if proyecto not in proyectos:
            print(f"  Phoenix no tiene el proyecto '{proyecto}'. Proyectos: {list(proyectos)}")
            return None
        if trazas < esperadas:
            print(f"  Atencion: Phoenix tiene {trazas} de {esperadas} trazas con esta etiqueta.")
        datos = (await gql(CONSULTA_METRICAS, {"id": proyectos[proyecto], "filtro": filtro}))["node"]

    total = datos["costSummary"]["total"]
    return {
        "filtro_en_phoenix": filtro,
        "trazas": trazas,
        "costo_total_usd": total["cost"],
        "costo_por_ejecucion_usd": (total["cost"] / trazas) if total["cost"] and trazas else None,
        "tokens_entrada": datos["costSummary"]["prompt"]["tokens"],
        "tokens_salida": datos["costSummary"]["completion"]["tokens"],
        "latencia_p50_s": datos["p50"] / 1000 if datos["p50"] else None,
        "latencia_p95_s": datos["p95"] / 1000 if datos["p95"] else None,
    }


async def contar_trazas(phoenix: httpx.AsyncClient, proyecto_id: str, filtro: str) -> int:
    """Cuantas trazas (spans raiz) tienen la etiqueta: una por ejecucion del grafo."""
    consulta = """query($id: ID!, $filtro: String) { node(id: $id) { ... on Project {
        spans(first: 100, filterCondition: $filtro) { edges { node { id } } } } } }"""
    filtro_raiz = f"{filtro} and parent_id is None"
    r = await phoenix.post("/graphql", json={"query": consulta, "variables": {"id": proyecto_id, "filtro": filtro_raiz}})
    r.raise_for_status()
    return len(r.json()["data"]["node"]["spans"]["edges"])
