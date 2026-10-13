"""Tests del RAG contra un servidor Chroma real, con embeddings simulados.

Necesitan Chroma levantado (docker compose up -d chroma). Si no responde, se
saltean en vez de fallar. Usan colecciones con prefijo "test" para no tocar
el indice real.
"""

from __future__ import annotations

import pytest

from comun.config import Configuracion
from rag.almacen import AlmacenChroma
from rag.embeddings import EmbeddingsSimulados
from rag.hibrido import BuscadorHibrido, filtro_metodologia_para, filtro_servicios_ofrecibles
from rag.indexar import asegurar_indice

CONFIG = Configuracion.para_pruebas()


@pytest.fixture
async def buscador():
    try:
        almacen = await AlmacenChroma.conectar(CONFIG)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Chroma no esta disponible: {error}")
    vectorizador = EmbeddingsSimulados()
    await asegurar_indice(CONFIG, almacen, vectorizador, forzar=True)
    return almacen, BuscadorHibrido(almacen, vectorizador, top_k=4)


async def test_indexado_idempotente(buscador):
    almacen, _ = buscador
    informe = await asegurar_indice(CONFIG, almacen, EmbeddingsSimulados())
    assert all(estado.startswith("sin cambios") for estado in informe.values())


async def test_busqueda_encuentra_storyline(buscador):
    _, b = buscador
    resultados = await b.buscar("catalogo", "localizar cursos de Storyline en arabe", filtro_servicios_ofrecibles())
    assert resultados[0].metadatos["servicio_id"] == "localizacion_storyline"


async def test_filtro_bloquea_servicios_en_evaluacion(buscador):
    _, b = buscador
    resultados = await b.buscar("catalogo", "avatares IA de Articulate", filtro_servicios_ofrecibles(), cantidad=20)
    assert all(r.metadatos["estado"] == "activo" for r in resultados)
    assert "avatares_ia" not in {r.metadatos["servicio_id"] for r in resultados}


async def test_qa_solo_aparece_si_se_piden_complementos(buscador):
    _, b = buscador
    sin = await b.buscar("catalogo", "QA de formato y testing", filtro_servicios_ofrecibles(), cantidad=20)
    con = await b.buscar("catalogo", "QA de formato y testing", filtro_servicios_ofrecibles(True), cantidad=20)
    assert "qa_formato" not in {r.metadatos["servicio_id"] for r in sin}
    assert con[0].metadatos["servicio_id"] == "qa_formato"


async def test_metodologia_filtrada_por_agente(buscador):
    _, b = buscador
    resultados = await b.buscar(
        "metodologia", "largo del primer email y asunto", filtro_metodologia_para("estratega_redactor")
    )
    assert resultados and all(r.metadatos["uso_estratega_redactor"] for r in resultados)
    assert resultados[0].metadatos["documento"] == "05_reglas_de_contacto"


async def test_fusion_registra_los_dos_puestos(buscador):
    _, b = buscador
    resultados = await b.buscar("catalogo", "DTP multilingue InDesign RTL", filtro_servicios_ofrecibles())
    primero = resultados[0]
    assert primero.metadatos["servicio_id"] == "dtp_multilingue"
    assert primero.puesto_vectorial is not None and primero.puesto_bm25 is not None
