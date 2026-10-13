"""Busqueda hibrida: vectores (significado) + BM25 (palabras exactas).

Por que dos busquedas:
    - La VECTORIAL entiende significado: "la agencia necesita alguien que
      maneje archivos .story" encuentra la ficha de Storyline aunque no diga
      "localizacion". Pero puede fallar con terminos tecnicos exactos.
    - BM25 (por palabras clave) es lo contrario: no entiende sinonimos, pero
      si la consulta dice "RTL" o "SCORM", encuentra exactamente esa palabra.
    Juntas cubren los puntos ciegos de cada una.

Como se combinan: Reciprocal Rank Fusion (RRF).
    Cada busqueda devuelve una lista ordenada. Cada fragmento suma
        1 / (60 + posicion)
    por cada lista en la que aparece. Se usan las POSICIONES y no los
    puntajes porque los puntajes de cada metodo estan en escalas distintas
    (una distancia coseno y un puntaje BM25 no se pueden sumar). Un fragmento
    que aparece arriba en las dos listas queda primero.

Filtros por metadatos (antes de buscar, en las DOS busquedas):
    Es la defensa contra alucinaciones del catalogo: si un servicio esta
    `en_evaluacion`, el agente directamente no lo ve, asi que no puede
    ofrecerlo. Es mas confiable que pedirle al LLM "no ofrezcas X".
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from rag.almacen import AlmacenChroma, FragmentoRecuperado
from rag.corpus import Coleccion
from rag.embeddings import Vectorizador

# Constante clasica de RRF: suaviza la diferencia entre el 1.o y el 2.o puesto.
K_RRF = 60

# Palabras demasiado comunes para aportar a BM25.
_VACIAS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo", "los",
    "para", "por", "que", "se", "su", "sus", "un", "una", "y", "o", "the", "and",
    "of", "to", "for", "in",
}


def _tokenizar(texto: str) -> list[str]:
    """Minusculas, sin acentos y sin palabras vacias (para BM25)."""
    sin_acentos = unicodedata.normalize("NFKD", texto.lower())
    limpio = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return [p for p in re.findall(r"\w+", limpio) if p not in _VACIAS]


class ResultadoBusqueda(BaseModel):
    """Un fragmento con el detalle de como llego al resultado (para la traza)."""

    id: str
    texto: str
    metadatos: dict[str, Any]
    puntaje_rrf: float
    puesto_vectorial: int | None = Field(default=None, description="1 = el mas cercano")
    puesto_bm25: int | None = Field(default=None, description="1 = mas palabras en comun")


# --------------------------------------------------------------------------
# Filtros listos para usar (sintaxis "where" de Chroma)
# --------------------------------------------------------------------------


def filtro_servicios_ofrecibles(incluir_complementos: bool = False) -> dict[str, Any]:
    """Solo servicios activos; los de prioridad 'complemento' (QA), solo si se piden."""
    if incluir_complementos:
        return {"estado": "activo"}
    return {"$and": [{"estado": "activo"}, {"prioridad": {"$ne": "complemento"}}]}


def filtro_metodologia_para(agente: str) -> dict[str, Any]:
    """Solo los documentos de metodologia que declaran ser para ese agente."""
    return {f"uso_{agente}": True}


# --------------------------------------------------------------------------
# Buscador
# --------------------------------------------------------------------------


class BuscadorHibrido:
    def __init__(self, almacen: AlmacenChroma, vectorizador: Vectorizador, top_k: int = 4) -> None:
        self._almacen = almacen
        self._vectorizador = vectorizador
        self._top_k = top_k

    async def _por_palabras(
        self, coleccion: Coleccion, consulta: str, filtro: dict[str, Any] | None
    ) -> list[FragmentoRecuperado]:
        """BM25 sobre los fragmentos que pasan el filtro, ordenados por puntaje.

        El indice BM25 se arma en el momento: el corpus tiene menos de 100
        fragmentos y armarlo toma milisegundos. Con un corpus grande se
        guardaria un indice disperso precalculado.
        """
        candidatos = await self._almacen.todos(coleccion, filtro)
        if not candidatos:
            return []
        bm25 = BM25Okapi([_tokenizar(c.texto) for c in candidatos])
        puntajes = bm25.get_scores(_tokenizar(consulta))
        ordenados = sorted(zip(candidatos, puntajes), key=lambda par: par[1], reverse=True)
        # Un puntaje 0 significa "ninguna palabra en comun": no es un resultado.
        return [c for c, p in ordenados if p > 0]

    async def buscar(
        self,
        coleccion: Coleccion,
        consulta: str,
        filtro: dict[str, Any] | None = None,
        cantidad: int | None = None,
    ) -> list[ResultadoBusqueda]:
        """Busqueda hibrida con fusion RRF. Devuelve los `cantidad` mejores."""
        cantidad = cantidad or self._top_k
        candidatos_por_lista = cantidad * 3  # cada metodo aporta mas candidatos que el final

        vector = await self._vectorizador.vectorizar_consulta(consulta)
        por_vector = await self._almacen.buscar_por_vector(
            coleccion, vector, candidatos_por_lista, filtro
        )
        por_palabras = (await self._por_palabras(coleccion, consulta, filtro))[:candidatos_por_lista]

        fusion: dict[str, ResultadoBusqueda] = {}
        for puesto, frag in enumerate(por_vector, start=1):
            fusion[frag.id] = ResultadoBusqueda(
                id=frag.id, texto=frag.texto, metadatos=frag.metadatos,
                puntaje_rrf=1 / (K_RRF + puesto), puesto_vectorial=puesto,
            )
        for puesto, frag in enumerate(por_palabras, start=1):
            if frag.id in fusion:
                fusion[frag.id].puntaje_rrf += 1 / (K_RRF + puesto)
                fusion[frag.id].puesto_bm25 = puesto
            else:
                fusion[frag.id] = ResultadoBusqueda(
                    id=frag.id, texto=frag.texto, metadatos=frag.metadatos,
                    puntaje_rrf=1 / (K_RRF + puesto), puesto_bm25=puesto,
                )

        ranking = sorted(fusion.values(), key=lambda r: r.puntaje_rrf, reverse=True)
        return ranking[:cantidad]

    async def buscar_diversificado(
        self,
        coleccion: Coleccion,
        consulta: str,
        filtro: dict[str, Any] | None = None,
        cantidad: int | None = None,
        clave: str = "servicio_id",
    ) -> list[ResultadoBusqueda]:
        """Como `buscar`, pero con un solo resultado (el mejor) por cada valor de `clave`.

        Diversificacion de resultados: una busqueda comun puede devolver cuatro
        fragmentos del mismo servicio. El especialista necesita OPCIONES
        distintas, asi que se piden mas candidatos y se queda con el mejor
        fragmento de cada servicio.
        """
        cantidad = cantidad or self._top_k
        candidatos = await self.buscar(coleccion, consulta, filtro, cantidad=cantidad * 4)
        vistos: set[Any] = set()
        diversos: list[ResultadoBusqueda] = []
        for r in candidatos:
            valor = r.metadatos.get(clave)
            if valor in vistos:
                continue
            vistos.add(valor)
            diversos.append(r)
            if len(diversos) == cantidad:
                break
        return diversos
