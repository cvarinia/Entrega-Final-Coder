"""Datos y guiones compartidos por los tests offline.

Un "guion" es la lista de respuestas que va a dar el modelo simulado, en
orden: cada AIMessage con tool_calls simula que el agente pidio usar una
herramienta; el ultimo AIMessage sin tool_calls es su resumen final. Asi se
prueba TODO el cableado (herramientas reales, validaciones reales, grafo
real, checkpointer real) sin gastar una llamada a la API.

La agencia "Acme Localization" es ficticia.
"""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.messages import AIMessage

from comun.web import WebSimulada

SITIO = "https://acme-loc.example"
VENDORS = "https://acme-loc.example/vendors"
ELEARNING = "https://acme-loc.example/elearning"

WEB = {
    SITIO: ("Acme Localization", "Acme Localization translates into 40 languages for life sciences and tech."),
    VENDORS: ("Join our vendor network", "We are looking for DTP and multimedia vendors. Apply through this form."),
    ELEARNING: ("E-learning localization", "We localize Articulate Storyline and Rise courses into 40+ languages."),
}

_ids = itertools.count(1)


def llamada(herramienta: str, **argumentos: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": argumentos, "id": f"call_{next(_ids)}"}])


def final(texto: str) -> AIMessage:
    return AIMessage(content=texto)


def web_simulada() -> WebSimulada:
    return WebSimulada(dict(WEB))


PERFIL = {
    "nombre": "Acme Localization",
    "sitio_web": SITIO,
    "resumen": "LSP con 40 idiomas, foco en life sciences y tecnologia, con servicio de e-learning.",
    "verticales": ["life sciences", "tecnologia"],
    "senales": [
        {"criterio": "servicio_elearning", "descripcion": "Tiene una pagina de localizacion de e-learning.", "fuente_url": ELEARNING},
        {"criterio": "trabaja_con_articulate", "descripcion": "Menciona cursos de Storyline y Rise.", "fuente_url": ELEARNING},
        {"criterio": "recluta_proveedores", "descripcion": "Busca proveedores de DTP y multimedia.", "fuente_url": VENDORS},
        {"criterio": "muchos_idiomas", "descripcion": "Trabaja con mas de 40 idiomas.", "fuente_url": SITIO},
    ],
    "canales_contacto": [
        {"tipo": "portal_proveedores", "descripcion": "Formulario Join our vendor network", "url": VENDORS},
    ],
}


def investigacion(perfil: dict | None = None) -> list[AIMessage]:
    return [
        llamada("buscar_web", consulta="Acme Localization e-learning vendors"),
        llamada("leer_pagina", url=VENDORS),
        llamada("entregar_perfil", perfil=perfil or PERFIL),
        final("Investigue a Acme: 4 senales, puntaje 9."),
    ]


def especialidad(servicios: list[tuple[str, str]] | None = None) -> list[AIMessage]:
    servicios = servicios or [
        ("localizacion_storyline", "Responde a sus cursos de Storyline en 40 idiomas."),
        ("dtp_multilingue", "Responde a su busqueda de proveedores de DTP."),
    ]
    return [
        llamada("buscar_servicios", necesidad="agencia que localiza cursos Storyline a muchos idiomas"),
        llamada("entregar_match",
                servicios=[{"servicio_id": s, "motivo": m} for s, m in servicios],
                argumento_central="Un solo proveedor para cursos y documentos en todos sus idiomas."),
        final("Elegi " + ", ".join(s for s, _ in servicios) + "."),
    ]


MENSAJE_OK = (
    "Hello Acme team, I saw that you localize Articulate Storyline and Rise courses into more than 40 "
    "languages and that you are looking for DTP and multimedia vendors. Headway handles the technical side "
    "of multilingual e-learning: Storyline localization, including right-to-left languages, and DTP for the "
    "related documents, as a single vendor. We have delivered more than 1,600 courses since 2010. Our rate "
    "is 14 USD/h. Would it be useful if we completed a short test with one of your files?"
)


def propuesta(mensaje: str = MENSAJE_OK) -> dict[str, Any]:
    return {
        "canal": "portal_proveedores",
        "motivo_canal": "Acme tiene un formulario publico para nuevos proveedores.",
        "idioma": "ingles",
        "asunto": "",
        "mensaje": mensaje,
        "piezas": ["brochure", "hoja_tarifas", "oferta_prueba"],
        "seguimiento": "Si responden, enviar la carta de presentacion y coordinar la prueba.",
    }


def redaccion(mensajes: list[str] | None = None) -> list[AIMessage]:
    """Cada mensaje es un intento de entrega (para probar la autocorreccion)."""
    intentos = [llamada("entregar_propuesta", propuesta=propuesta(m)) for m in (mensajes or [MENSAJE_OK])]
    return [llamada("consultar_metodologia", consulta="reglas del primer contacto"), *intentos,
            final("Propuesta lista por el portal de proveedores.")]
