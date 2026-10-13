"""Reglas de negocio que se verifican con CODIGO, no con el LLM.

Salen del corpus de metodologia (05_reglas_de_contacto y 06_reglas_de_precio).
El LLM las lee via RAG para saber COMO escribir; este modulo verifica que lo
haya hecho. Se usan en dos lugares:
    - en la herramienta `entregar_propuesta`, para que el redactor corrija
      en el momento (el error vuelve como respuesta de la herramienta);
    - en los chequeos automaticos que ve el Supervisor (defensa en
      profundidad: si algo se escapo, el Supervisor lo manda a corregir).

La regla estrella: TODO precio que aparezca en el mensaje tiene que ser una
tarifa de un servicio aprobado. Un LLM al que le falta un numero tiende a
inventar uno razonable; en una propuesta comercial eso es grave.
"""

from __future__ import annotations

import re

from state import MatchServicios, Propuesta

# Limites de largo por canal (ver 05_reglas_de_contacto.md).
PALABRAS_EMAIL = (50, 125)
PALABRAS_PORTAL = (50, 200)        # campo libre de un formulario de alta
CARACTERES_LINKEDIN = 300
PALABRAS_ASUNTO = (4, 7)

# "14 USD", "USD 14", "US$14", "$14", "14 dolares", "14 USD/h"...
_PATRON_PRECIO = re.compile(
    r"(?:(?:US\$|USD|U\$S|\$|€|EUR)\s?(\d+(?:[.,]\d+)?))"
    r"|(?:(\d+(?:[.,]\d+)?)\s?(?:USD|US\$|dólares|dolares|dollars|euros?|EUR|€))",
    re.IGNORECASE,
)

# Palabras que indican que se ofrece una prueba, en los idiomas mas probables.
_PALABRAS_PRUEBA = (
    "test", "sample", "trial", "pilot", "prueba", "muestra", "teste", "amostra",
    "essai", "échantillon", "probe",
)

# Frases que contradicen el posicionamiento de Headway (estudio pequeno, cercano y
# confiable). Salieron de la revision de Carla de la primera propuesta real.
_PATRON_EXAGERADO = re.compile(
    r"exactly what|\bscal(?:e|es|ing|able)\b|large[- ]scale|world[- ]class|\bleading\b|"
    r"\bthe best\b|best[- ]in[- ]class|\bperfect(?:ly)?\b|cutting[- ]edge|"
    r"exactamente lo que|\bescalar\b|a gran escala|\bl[ií]der(?:es)?\b|el mejor|perfect[oa]",
    re.IGNORECASE,
)

# Palabras que indican que la persona pidio QA (servicio "complemento").
_PATRON_PIDE_COMPLEMENTO = re.compile(r"\b(qa|testing|tests?|control de calidad|quality)\b", re.IGNORECASE)


def contar_palabras(texto: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", texto))


def precios_en_texto(texto: str) -> list[str]:
    """Numeros que aparecen como precio en el texto, normalizados ('14', '12.5')."""
    numeros = []
    for coincidencia in _PATRON_PRECIO.finditer(texto):
        numero = coincidencia.group(1) or coincidencia.group(2)
        numeros.append(numero.replace(",", "."))
    return numeros


def numeros_de_tarifa(tarifa: str) -> set[str]:
    """'14 USD/h' -> {'14'}. 'pedir cotización' -> vacio."""
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", tarifa)}


def ofrece_prueba(texto: str) -> bool:
    minusculas = texto.lower()
    return any(palabra in minusculas for palabra in _PALABRAS_PRUEBA)


def pide_complementos(pedido: str) -> bool:
    """True si la persona menciono QA/testing: solo ahi se puede proponer un complemento."""
    return bool(_PATRON_PIDE_COMPLEMENTO.search(pedido))


def frases_exageradas(texto: str) -> list[str]:
    """Frases que contradicen el posicionamiento (estudio pequeno, cercano y confiable)."""
    return sorted({c.group(0).lower() for c in _PATRON_EXAGERADO.finditer(texto)})


def revisar_propuesta(propuesta: Propuesta, match: MatchServicios | None) -> list[str]:
    """Devuelve la lista de problemas (vacia = la propuesta cumple todas las reglas)."""
    problemas: list[str] = []
    mensaje = propuesta.mensaje

    # --- Largo segun el canal ---------------------------------------------
    if propuesta.canal == "linkedin":
        if len(mensaje) > CARACTERES_LINKEDIN:
            problemas.append(
                f"El mensaje de LinkedIn tiene {len(mensaje)} caracteres; el maximo es {CARACTERES_LINKEDIN}."
            )
    else:
        minimo, maximo = PALABRAS_EMAIL if propuesta.canal == "email" else PALABRAS_PORTAL
        palabras = contar_palabras(mensaje)
        if not minimo <= palabras <= maximo:
            problemas.append(
                f"El mensaje tiene {palabras} palabras; para el canal '{propuesta.canal}' "
                f"tiene que tener entre {minimo} y {maximo}."
            )

    # --- Asunto (obligatorio salvo en LinkedIn) -----------------------------
    if propuesta.canal == "email":
        palabras_asunto = contar_palabras(propuesta.asunto)
        minimo, maximo = PALABRAS_ASUNTO
        if not minimo <= palabras_asunto <= maximo:
            problemas.append(
                f"El asunto tiene {palabras_asunto} palabras; tiene que tener entre {minimo} y {maximo}."
            )

    # --- Oferta de prueba -----------------------------------------------------
    if not ofrece_prueba(mensaje) or "oferta_prueba" not in propuesta.piezas:
        problemas.append(
            "El primer mensaje tiene que ofrecer una prueba (test/sample) y 'oferta_prueba' "
            "tiene que estar en las piezas."
        )

    # --- Tono: nada de exageraciones ------------------------------------------
    exageradas = frases_exageradas(mensaje)
    if exageradas:
        problemas.append(
            f"Tono exagerado: {exageradas}. Headway es un estudio pequeno, cercano y confiable; "
            "escribi en ese tono, sin superlativos ni promesas de escala."
        )

    # --- Precios: solo tarifas de servicios aprobados -----------------------
    permitidos: set[str] = set()
    for servicio in (match.servicios if match else []):
        permitidos |= numeros_de_tarifa(servicio.tarifa)
    inventados = [p for p in precios_en_texto(mensaje) if p not in permitidos]
    if inventados:
        problemas.append(
            f"El mensaje menciona precios que no son tarifas de los servicios aprobados: {inventados}. "
            f"Tarifas permitidas: {sorted(permitidos) or 'ninguna (escribir pedir cotizacion)'}."
        )

    return problemas
