"""Tests de las reglas de negocio de la propuesta (sin Chroma ni API)."""

from __future__ import annotations

from agents.reglas import contar_palabras, ofrece_prueba, pide_complementos, precios_en_texto, revisar_propuesta
from state import MatchServicios, Propuesta, ServicioPropuesto
from tests import guiones as g

MATCH = MatchServicios(
    servicios=[
        ServicioPropuesto(servicio_id="localizacion_storyline", motivo="Cursos de Storyline.",
                          servicio="Storyline", prioridad="estrella", tarifa="14 USD/h"),
        ServicioPropuesto(servicio_id="creacion_cursos", motivo="Desarrollo de cursos.",
                          servicio="Creacion", prioridad="estrella", tarifa="pedir cotización"),
    ],
    argumento_central="Un solo proveedor para todo el proyecto.",
)


def propuesta(**cambios) -> Propuesta:
    return Propuesta(**{**g.propuesta(), **cambios})


def test_detecta_precios_en_varios_formatos():
    assert precios_en_texto("Our rate is 14 USD/h, or US$ 20 per hour, or $12.5, or 30 dollars") == ["14", "20", "12.5", "30"]
    assert precios_en_texto("We have 40 languages and 1,600 courses") == []


def test_propuesta_valida():
    assert revisar_propuesta(propuesta(), MATCH) == []


def test_precio_que_no_esta_en_el_catalogo():
    problemas = revisar_propuesta(propuesta(mensaje=g.MENSAJE_OK.replace("14 USD/h", "20 USD/h")), MATCH)
    assert any("20" in p for p in problemas)


def test_email_demasiado_largo_y_asunto_corto():
    largo = g.MENSAJE_OK + " " + "More details about our team and tools. " * 10
    problemas = revisar_propuesta(propuesta(canal="email", asunto="Hola", mensaje=largo), MATCH)
    assert any("palabras" in p and "entre 50 y 125" in p for p in problemas)
    assert any("asunto" in p for p in problemas)


def test_sin_oferta_de_prueba():
    sin_prueba = g.MENSAJE_OK.replace("Would it be useful if we completed a short test with one of your files?",
                                      "Would you like to talk?")
    assert any("prueba" in p for p in revisar_propuesta(propuesta(mensaje=sin_prueba), MATCH))


def test_linkedin_limite_de_caracteres():
    problemas = revisar_propuesta(propuesta(canal="linkedin", mensaje=g.MENSAJE_OK), MATCH)
    assert any("caracteres" in p for p in problemas)


def test_auxiliares():
    assert contar_palabras("Storyline localization support for Acme") == 5
    assert ofrece_prueba("We can do a short TEST") and not ofrece_prueba("Let's talk")
    assert pide_complementos("ofrecele QA") and not pide_complementos("ofrecele Storyline")


def test_tono_exagerado_se_rechaza():
    """Regresion de la revision de Carla: 'exactly what Vistatec needs to scale...'."""
    exagerado = g.MENSAJE_OK.replace("Hello Acme team,", "Headway is exactly what Acme needs to scale. Hello,")
    problemas = revisar_propuesta(propuesta(mensaje=exagerado), MATCH)
    assert any("Tono exagerado" in p and "exactly what" in p for p in problemas)


def test_cierre_aprobado_por_headway_pasa():
    cierre = ("With over 10 years in e-learning localization, we work as a single point of contact for your team, "
              "reducing coordination overhead. Would a short test project in Storyline or Rise help demonstrate our fit?")
    mensaje = "Hello Acme team, I saw that you localize Storyline and Rise courses and that you are recruiting DTP vendors. " + cierre
    assert revisar_propuesta(propuesta(mensaje=mensaje), MATCH) == []
