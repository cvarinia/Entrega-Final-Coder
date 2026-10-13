"""Tests del chunking y los metadatos del corpus (no necesitan Chroma ni API)."""

from __future__ import annotations

from comun.config import Configuracion
from rag.corpus import SIN_TARIFA, cargar_corpus, huella, partir_por_secciones, separar_frontmatter

CONFIG = Configuracion.para_pruebas()


async def test_carga_las_dos_colecciones():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    assert set(corpus) == {"catalogo", "metodologia"}
    assert len(corpus["catalogo"]) > 10
    assert len(corpus["metodologia"]) > 10


async def test_ids_unicos_y_estables():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    ids = [f.id for frags in corpus.values() for f in frags]
    assert len(ids) == len(set(ids))
    assert "catalogo:01_localizacion_storyline:01" in ids


async def test_metadatos_del_catalogo():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    por_servicio = {f.metadatos["servicio_id"]: f.metadatos for f in corpus["catalogo"]}
    assert por_servicio["localizacion_storyline"]["tarifa"] == "14 USD/h"
    assert por_servicio["localizacion_storyline"]["prioridad"] == "estrella"
    assert por_servicio["avatares_ia"]["estado"] == "en_evaluacion"
    assert por_servicio["qa_formato"]["prioridad"] == "complemento"
    # Una ficha sin tarifa (null) queda como "pedir cotizacion", nunca vacia.
    assert por_servicio["localizacion_figma"]["tarifa"] == SIN_TARIFA
    assert por_servicio["creacion_cursos"]["tarifa"] == SIN_TARIFA


async def test_metadatos_de_metodologia_por_agente():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    precio = [f for f in corpus["metodologia"] if f.documento == "06_reglas_de_precio"]
    assert precio and all(f.metadatos["uso_estratega_redactor"] for f in precio)
    assert not any(f.metadatos["uso_investigador"] for f in precio)


async def test_cada_fragmento_lleva_encabezado_de_contexto():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    for f in corpus["catalogo"]:
        assert f.texto.startswith(f"Servicio: {f.metadatos['servicio']}")
    for f in corpus["metodologia"]:
        assert f.texto.startswith("Documento: ")


def test_secciones_de_nivel_3_heredan_el_padre():
    _, secciones = partir_por_secciones("# T\n## Padre\nintro\n### Hijo\ntexto")
    assert [s for s, _ in secciones] == ["Padre", "Padre > Hijo"]


def test_frontmatter():
    datos, cuerpo = separar_frontmatter("---\nestado: activo\n---\n# Hola")
    assert datos == {"estado": "activo"} and cuerpo.strip() == "# Hola"


async def test_huella_cambia_con_el_modelo_de_embeddings():
    corpus = await cargar_corpus(CONFIG.ruta_corpus)
    frags = corpus["catalogo"]
    assert huella(frags, "modelo-a") == huella(frags, "modelo-a")
    assert huella(frags, "modelo-a") != huella(frags, "modelo-b")
