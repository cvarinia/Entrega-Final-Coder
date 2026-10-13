"""Human-in-the-loop: una persona aprueba los servicios antes de redactar.

Por que aca: el match de servicios es la decision comercial del sistema (que
se le ofrece a la agencia). Es barato corregirlo antes de redactar y caro
despues. Ademas, Carla conoce a su mercado: el sistema propone y ella decide.

Como funciona la pausa (igual que en la PE7):
    1. `interrupt(pedido)` guarda el estado en el checkpointer y corta la
       ejecucion. El worker queda libre para otros jobs.
    2. La persona responde por POST /tasks/{id}/approve (o en la CLI).
    3. Se reanuda con `Command(resume=respuesta)` y el MISMO thread_id;
       `interrupt()` devuelve esa respuesta y el nodo sigue.

Regla de LangGraph: al reanudar, el nodo que llamo a interrupt() se vuelve a
ejecutar DESDE EL PRINCIPIO. Por eso aca no hay nada con efectos secundarios
antes de la pausa: solo se arma la pregunta.

Si la persona rechaza (con un comentario), el grafo vuelve al Supervisor, que
manda al especialista a corregir. El match nuevo vuelve a pedir aprobacion:
una aprobacion vale para UNA version del match.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from state import DecisionHumana, EstadoPropuesta

ACCION_CRITICA = "aprobar_servicios"


def armar_pedido_aprobacion(estado: EstadoPropuesta) -> dict[str, Any]:
    """Lo que ve la persona. Tiene que ser JSON: viaja al checkpointer y a la API."""
    perfil, match = estado.get("perfil"), estado.get("match")
    return {
        "accion": ACCION_CRITICA,
        "pregunta": "¿Aprobas estos servicios para la propuesta? Si no, deja un comentario con que cambiar.",
        "version_match": estado.get("version_match", 0),
        "lsp": {
            "nombre": perfil.nombre if perfil else None,
            "puntaje": perfil.puntaje if perfil else None,
            "prioridad_contacto": perfil.prioridad_contacto if perfil else None,
            "senales": [f"{s.criterio}: {s.descripcion} ({s.fuente_url})" for s in perfil.senales] if perfil else [],
        },
        "servicios": [s.model_dump() for s in match.servicios] if match else [],
        "argumento_central": match.argumento_central if match else "",
    }


def crear_nodo_aprobacion():
    async def nodo_aprobacion(estado: EstadoPropuesta) -> dict[str, Any]:
        pedido = armar_pedido_aprobacion(estado)
        print(f"  [aprobacion] pausa: esperando decision humana (match v{pedido['version_match']})", flush=True)

        # --- Aca se detiene el grafo. -----------------------------------
        respuesta = interrupt(pedido)

        # Al reanudar: validacion del contrato (si no cumple -> ValidationError -> FAILED).
        decision = DecisionHumana.model_validate(respuesta)
        veredicto = "APROBADO" if decision.aprobado else "RECHAZADO"
        print(f"  [aprobacion] {veredicto} por {decision.revisor}", flush=True)

        actualizacion: dict[str, Any] = {
            "decision_match": decision,
            "messages": [AIMessage(
                content=f"Match {veredicto.lower()} por {decision.revisor}. {decision.comentario}".strip(),
                name="aprobacion",
            )],
        }
        if decision.aprobado:
            actualizacion["version_aprobada"] = estado.get("version_match", 0)
        return actualizacion

    return nodo_aprobacion


def rutear_tras_aprobacion(estado: EstadoPropuesta) -> str:
    """Aprobado -> redactor. Rechazado -> vuelve al Supervisor para corregir el match."""
    decision = estado.get("decision_match")
    return "redactor" if decision is not None and decision.aprobado else "supervisor"


def pedido_pendiente(resultado: dict[str, Any]) -> dict[str, Any] | None:
    """Si el grafo quedo pausado, devuelve lo que se le pregunta a la persona."""
    interrupciones = resultado.get("__interrupt__") or []
    return interrupciones[0].value if interrupciones else None
