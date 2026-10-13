"""Nodo de sintesis: arma la respuesta final a partir de los datos VALIDADOS.

Diferencia con la PE6/PE7: aca la sintesis NO usa el LLM. Es una plantilla.

Por que: el mensaje para la LSP ya paso por las reglas (largo, asunto, prueba,
precios del catalogo) y fue escrito sobre un match aprobado por una persona.
Si un LLM lo "resumiera" o "mejorara" al final, podria cambiar una palabra,
un precio o el tono, y todas esas validaciones dejarian de valer. Con una
plantilla, lo que se valido es exactamente lo que se entrega. Ademas, es
gratis y no puede fallar por un error del proveedor.

Si la tarea no se completo, la respuesta lo dice al principio y explica que
falto, en vez de disimularlo.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from state import EstadoPropuesta

NOMBRES_PIEZAS = {
    "brochure": "Brochure de servicios",
    "carta_presentacion": "Carta de presentación (para el seguimiento)",
    "hoja_tarifas": "Hoja de tarifas",
    "oferta_prueba": "Ofrecimiento de una prueba (test)",
}


def armar_respuesta(estado: EstadoPropuesta) -> str:
    perfil, match, propuesta = estado.get("perfil"), estado.get("match"), estado.get("propuesta")
    partes: list[str] = []

    if not estado.get("task_completed"):
        partes.append(
            "⚠️ **La propuesta NO quedó completa.** "
            f"{estado.get('motivo_cierre', '')}\n\nAbajo va lo que sí se pudo validar."
        )

    if perfil:
        senales = "\n".join(f"- **{s.criterio}**: {s.descripcion} ([fuente]({s.fuente_url}))" for s in perfil.senales)
        canales = "\n".join(
            f"- {c.tipo}: {c.descripcion}" + (f" ({c.url})" if c.url else "") for c in perfil.canales_contacto
        ) or "- (no se encontraron canales públicos)"
        partes.append(
            f"## 1. La agencia: {perfil.nombre}\n"
            f"{perfil.resumen}\n\n"
            f"**Puntaje:** {perfil.puntaje}/11 → prioridad de contacto **{perfil.prioridad_contacto}**  \n"
            f"**Sitio:** {perfil.sitio_web}\n\n"
            f"**Señales encontradas:**\n{senales}\n\n**Canales de contacto:**\n{canales}"
        )

    if match:
        filas = "\n".join(
            f"| {s.servicio}{' (adicional)' if s.tipo == 'add_on' else ''} | {s.prioridad} | {s.tarifa} | {s.motivo} |"
            for s in match.servicios
        )
        decision = estado.get("decision_match")
        aprobado = estado.get("version_aprobada", 0) == estado.get("version_match", 0) and decision is not None
        partes.append(
            "## 2. Servicios a ofrecer\n"
            f"{match.argumento_central}\n\n"
            "| Servicio | Prioridad | Tarifa | Por qué |\n|---|---|---|---|\n" + filas + "\n\n"
            + (f"✅ Aprobado por **{decision.revisor}**." if aprobado else "⏳ Sin aprobación humana.")
        )

    if propuesta:
        piezas = "\n".join(f"- {NOMBRES_PIEZAS.get(p, p)}" for p in propuesta.piezas)
        asunto = f"**Asunto:** {propuesta.asunto}\n\n" if propuesta.asunto else ""
        partes.append(
            f"## 3. Cómo contactarla\n"
            f"**Canal:** {propuesta.canal} — {propuesta.motivo_canal}\n\n"
            f"**Piezas a enviar:**\n{piezas}\n\n"
            f"## 4. Mensaje listo para enviar ({propuesta.idioma})\n"
            f"{asunto}```\n{propuesta.mensaje}\n```\n\n"
            f"## 5. Si responde\n{propuesta.seguimiento}"
        )

    return "\n\n".join(partes) or "No se pudo avanzar con el pedido."


def crear_nodo_sintesis():
    async def nodo_sintesis(estado: EstadoPropuesta) -> dict[str, Any]:
        print("  [sintesis] armando la respuesta final (plantilla, sin LLM)...", flush=True)
        return {"messages": [AIMessage(content=armar_respuesta(estado), name="sintesis")]}

    return nodo_sintesis
