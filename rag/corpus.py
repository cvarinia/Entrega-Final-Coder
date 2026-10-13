"""Lectura y fragmentacion (chunking) del corpus de Headway.

El corpus son archivos Markdown en data/corpus/, en dos colecciones:

    catalogo/     una ficha por servicio  -> la usa el Especialista de servicios
    metodologia/  como venderle a una LSP -> la usan el Investigador y el Redactor

Cada archivo tiene dos partes, y cada una cumple un rol distinto en el RAG:

    ---                          <- FRONTMATTER (YAML): se convierte en METADATOS.
    prioridad: estrella             No se vectoriza: se usa para FILTRAR antes de
    estado: activo                  buscar ("solo servicios activos").
    tarifa: "14 USD/h"
    ---
    # Titulo                     <- CUERPO: se parte en fragmentos por seccion y
    ## Seccion                      cada fragmento se convierte en un vector.

Estrategia de chunking: un fragmento por seccion (## o ###), no por cantidad
de caracteres. El corpus se escribio para que cada seccion se entienda sola,
asi que cortar por titulos respeta las unidades de significado.

Ademas, cada fragmento lleva un ENCABEZADO DE CONTEXTO con el documento y la
seccion de donde sale ("Servicio: Localizacion Storyline | Seccion: Precio").
Sin eso, un fragmento que solo dice "14 USD/h" no sabria de que servicio es.

Este modulo no sabe nada de embeddings ni de Chroma: solo lee y parte texto.
Leer archivos es I/O bloqueante, por eso `cargar_corpus` lo manda a un hilo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Coleccion = Literal["catalogo", "metodologia"]
COLECCIONES: tuple[Coleccion, ...] = ("catalogo", "metodologia")

# Agentes que pueden figurar en `usado_por` (metodologia). Chroma no guarda
# listas como metadato, asi que cada agente se vuelve un booleano: uso_redactor=True.
AGENTES = ("investigador", "especialista_servicios", "estratega_redactor", "supervisor")

# Lo que se escribe cuando una ficha no tiene tarifa. Es la regla de negocio:
# el sistema nunca estima un precio.
SIN_TARIFA = "pedir cotización"

_PATRON_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_PATRON_TITULO = re.compile(r"^(#{1,3})\s+(.+?)\s*$")

TipoMetadato = str | int | float | bool


class Fragmento(BaseModel):
    """Un pedazo del corpus listo para indexar."""

    id: str = Field(description="Identificador estable: coleccion:documento:numero")
    coleccion: Coleccion
    documento: str = Field(description="Nombre del archivo sin extension")
    seccion: str = Field(description="Ruta de titulos, por ejemplo 'Precio' o 'Los cuatro factores > Tiempo'")
    texto: str = Field(description="Encabezado de contexto + contenido de la seccion")
    metadatos: dict[str, TipoMetadato]


# --------------------------------------------------------------------------
# Lectura de un archivo
# --------------------------------------------------------------------------


def separar_frontmatter(contenido: str) -> tuple[dict, str]:
    """Devuelve (frontmatter como dict, cuerpo en Markdown)."""
    coincidencia = _PATRON_FRONTMATTER.match(contenido)
    if not coincidencia:
        return {}, contenido
    datos = yaml.safe_load(coincidencia.group(1)) or {}
    return datos, contenido[coincidencia.end():]


def partir_por_secciones(cuerpo: str) -> tuple[str, list[tuple[str, str]]]:
    """Parte el cuerpo por titulos ## y ###.

    Devuelve (titulo del documento, [(ruta de la seccion, texto), ...]).
    Un ### hereda el nombre de su ## padre: "Los cuatro factores > 1. Resultado".
    """
    titulo_documento = ""
    secciones: list[tuple[str, str]] = []
    padre = ""
    ruta_actual = "Introducción"
    lineas_actuales: list[str] = []

    def cerrar() -> None:
        texto = "\n".join(lineas_actuales).strip()
        if texto:
            secciones.append((ruta_actual, texto))

    for linea in cuerpo.splitlines():
        coincidencia = _PATRON_TITULO.match(linea)
        if not coincidencia:
            lineas_actuales.append(linea)
            continue

        nivel, titulo = len(coincidencia.group(1)), coincidencia.group(2)
        if nivel == 1:
            titulo_documento = titulo
            continue
        cerrar()
        lineas_actuales = []
        if nivel == 2:
            padre = titulo
            ruta_actual = titulo
        else:  # nivel 3
            ruta_actual = f"{padre} > {titulo}" if padre else titulo

    cerrar()
    return titulo_documento, secciones


# --------------------------------------------------------------------------
# Metadatos por coleccion
# --------------------------------------------------------------------------


def _metadatos_catalogo(frontmatter: dict) -> dict[str, TipoMetadato]:
    tarifa = frontmatter.get("tarifa")
    return {
        "servicio": str(frontmatter.get("servicio", "")),
        "servicio_id": str(frontmatter.get("id", "")),
        "prioridad": str(frontmatter.get("prioridad", "")),
        "estado": str(frontmatter.get("estado", "")),
        "tipo": str(frontmatter.get("tipo", "servicio")),
        # Sin tarifa (null) o "pedir cotizacion": las dos significan lo mismo.
        "tarifa": str(tarifa).strip() if tarifa else SIN_TARIFA,
        "herramientas": ", ".join(frontmatter.get("herramientas") or []),
    }


def _metadatos_metodologia(frontmatter: dict) -> dict[str, TipoMetadato]:
    usado_por = set(frontmatter.get("usado_por") or [])
    return {f"uso_{agente}": agente in usado_por for agente in AGENTES}


def _encabezado(coleccion: Coleccion, titulo: str, seccion: str, meta: dict) -> str:
    if coleccion == "catalogo":
        return (
            f"Servicio: {meta['servicio']} (prioridad: {meta['prioridad']}, "
            f"tarifa: {meta['tarifa']}) | Sección: {seccion}"
        )
    return f"Documento: {titulo} | Sección: {seccion}"


def fragmentar_archivo(ruta: Path, coleccion: Coleccion) -> list[Fragmento]:
    """Lee un archivo del corpus y lo devuelve partido en fragmentos."""
    frontmatter, cuerpo = separar_frontmatter(ruta.read_text(encoding="utf-8"))
    titulo, secciones = partir_por_secciones(cuerpo)
    documento = ruta.stem

    base = (
        _metadatos_catalogo(frontmatter) if coleccion == "catalogo"
        else _metadatos_metodologia(frontmatter)
    )

    fragmentos: list[Fragmento] = []
    for numero, (seccion, texto) in enumerate(secciones, start=1):
        metadatos = {
            **base,
            "coleccion": coleccion,
            "documento": documento,
            "titulo": titulo,
            "seccion": seccion,
        }
        fragmentos.append(
            Fragmento(
                id=f"{coleccion}:{documento}:{numero:02d}",
                coleccion=coleccion,
                documento=documento,
                seccion=seccion,
                texto=f"{_encabezado(coleccion, titulo, seccion, metadatos)}\n\n{texto}",
                metadatos=metadatos,
            )
        )
    return fragmentos


# --------------------------------------------------------------------------
# Corpus completo
# --------------------------------------------------------------------------


def _leer_coleccion(ruta_corpus: Path, coleccion: Coleccion) -> list[Fragmento]:
    carpeta = ruta_corpus / coleccion
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del corpus: {carpeta}")
    fragmentos: list[Fragmento] = []
    for ruta in sorted(carpeta.glob("*.md")):
        fragmentos.extend(fragmentar_archivo(ruta, coleccion))
    return fragmentos


async def cargar_corpus(ruta_corpus: Path) -> dict[Coleccion, list[Fragmento]]:
    """Lee las dos colecciones sin bloquear el event loop."""
    return {
        coleccion: await asyncio.to_thread(_leer_coleccion, ruta_corpus, coleccion)
        for coleccion in COLECCIONES
    }


def huella(fragmentos: list[Fragmento], modelo_embeddings: str) -> str:
    """Hash del contenido + el modelo de embeddings.

    Sirve para no re-indexar si nada cambio. Incluye el modelo a proposito:
    si cambia el modelo de embeddings, los vectores viejos no sirven (cada
    modelo arma su propio "mapa" de significados) y hay que re-indexar.
    """
    contenido = json.dumps(
        [modelo_embeddings] + [f.model_dump() for f in fragmentos],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:16]
