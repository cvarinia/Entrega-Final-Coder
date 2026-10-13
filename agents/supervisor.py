"""Nodo Supervisor: coordina, no ejecuta (patron de la PE6/PE7).

En cada ronda:
    1. Lee el pedido, el estado resumido (perfil, match, aprobacion, propuesta)
       y los chequeos automaticos.
    2. Decide con salida estructurada quien interviene o si es momento de FINALIZAR.
    3. Escribe una instruccion autocontenida para el agente elegido.

Lo que NO decide el Supervisor:
    - La aprobacion humana: si elige "redactor" y el match no esta aprobado,
      el grafo pasa primero por la aprobacion (ver graph.rutear). Un LLM no
      puede saltearse ese paso.
    - Si la tarea quedo completa: aunque elija FINALIZAR, la tarea solo cuenta
      como completa si los chequeos automaticos lo respaldan.
    - El limite de rondas: al llegarlo, el codigo cierra sin preguntarle al LLM.
    - El estancamiento: si un agente no pudo entregar dos veces seguidas, el
      codigo cierra y explica por que (ver agente_estancado).
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.validacion import chequeos_a_texto, chequeos_automaticos, pedido_original, todo_ok
from comun.config import Configuracion
from comun.recursos import Recursos
from comun.reintentos import con_reintentos
from state import Aporte, DecisionSupervisor, EstadoPropuesta, RegistroDecision

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo que arma propuestas comerciales de Headway
(localizacion de e-learning y DTP multilingue) para agencias de localizacion (LSPs).

Tu equipo:
- "investigador": investiga la LSP en internet y entrega su perfil (senales con fuentes y
  canales de contacto). No elige servicios ni redacta.
- "especialista": cruza el perfil con el catalogo de Headway y elige de 1 a 4 servicios.
- "redactor": con los servicios APROBADOS por una persona, elige canal y piezas y redacta
  el primer mensaje.

Tu trabajo es COORDINAR. Nunca respondas vos el pedido.

Orden del flujo y rubrica:
0. Si el pedido NO identifica una agencia concreta (un nombre real o un sitio web; un
   texto como "<nombre>" no cuenta), o un agente respondio que le falta un dato que solo
   puede dar la persona -> "FINALIZAR" de inmediato, y en la evaluacion explica que dato
   falta. No mandes a un agente a pedirle datos a la persona: los agentes no hablan con ella.
1. Sin perfil, o con un chequeo de perfil en FALLA (por ejemplo fuentes no consultadas)
   -> "investigador", diciendo exactamente que corregir.
2. Con perfil OK pero sin match vigente, o con un chequeo de match en FALLA, o con el
   match RECHAZADO por la persona -> "especialista". Si hubo rechazo, copia el comentario
   de la persona en la instruccion.
3. Con match vigente y sin propuesta vigente -> "redactor". (Si el match no esta aprobado,
   el sistema le pide la aprobacion a una persona antes de que el redactor empiece: vos
   igual elegi "redactor".)
4. Propuesta con algun chequeo en FALLA -> "redactor" con lo que hay que corregir.
5. Todos los chequeos en OK -> "FINALIZAR".

Las instrucciones tienen que ser AUTOCONTENIDAS: el agente no ve esta conversacion.
Nunca pidas telefonos, emails personales ni nombres de clientes de la agencia: el
sistema solo usa canales publicos de la empresa.
Inclui siempre el nombre de la LSP y lo que pidio la persona (por ejemplo el idioma
o un servicio que le interese ofrecer).
"""


def _estado_resumido(estado: EstadoPropuesta) -> str:
    lineas: list[str] = []
    perfil = estado.get("perfil")
    if perfil:
        lineas.append(
            f"PERFIL (ronda {estado.get('ronda_perfil')}): {perfil.nombre} | puntaje {perfil.puntaje}/11 | "
            f"{len(perfil.senales)} senales: " + ", ".join(sorted({s.criterio for s in perfil.senales}))
            + f" | canales: {[c.tipo for c in perfil.canales_contacto] or 'ninguno'}"
        )
    match = estado.get("match")
    if match:
        lineas.append(
            f"MATCH v{estado.get('version_match')} (ronda {estado.get('ronda_match')}): "
            + "; ".join(f"{s.servicio_id} ({s.prioridad})" for s in match.servicios)
        )
    propuesta = estado.get("propuesta")
    if propuesta:
        lineas.append(f"PROPUESTA (sobre match v{estado.get('version_match_propuesta')}): canal {propuesta.canal}, "
                      f"idioma {propuesta.idioma}")
    for aporte in estado.get("aportes", [])[-3:]:
        lineas.append(f"[ronda {aporte.ronda}] {aporte.agente}: {aporte.resumen}")
    return "\n".join(lineas) or "(todavia no intervino ningun agente)"


def agente_estancado(aportes: list[Aporte]) -> Aporte | None:
    """Detector de estancamiento: el mismo agente fallo dos veces seguidas sin entregar.

    Caso real de la primera prueba: el pedido no tenia el nombre de la agencia.
    El investigador respondio "falta el nombre" y el Supervisor lo volvio a
    mandar 8 veces, hasta el limite de rondas. Si un agente no pudo entregar
    dos veces seguidas, el problema no se arregla insistiendo: hay que cerrar y
    decirle a la persona que falta. Lo decide el codigo, no el LLM.
    """
    if len(aportes) < 2:
        return None
    anteultimo, ultimo = aportes[-2], aportes[-1]
    if anteultimo.agente == ultimo.agente and not anteultimo.entrego and not ultimo.entrego:
        return ultimo
    return None


def crear_nodo_supervisor(modelo: BaseChatModel, config: Configuracion, recursos: Recursos):
    decisor = modelo.with_structured_output(DecisionSupervisor)

    async def nodo_supervisor(estado: EstadoPropuesta) -> dict[str, Any]:
        ronda = estado.get("ronda", 0) + 1
        chequeos = await chequeos_automaticos(estado, recursos.catalogo)

        # --- Condiciones de parada duras: las decide el codigo ------------
        motivo_forzado = None
        if ronda > config.max_rondas_supervisor:
            motivo_forzado = f"Se alcanzo el maximo de {config.max_rondas_supervisor} rondas sin cumplir la rubrica."
        elif (estancado := agente_estancado(estado.get("aportes", []))) is not None:
            motivo_forzado = (
                f"El agente '{estancado.agente}' no pudo entregar dos veces seguidas; insistir no sirve. "
                f"Lo que dijo: {estancado.resumen}"
            )
        if motivo_forzado:
            registro = RegistroDecision(ronda=ronda, siguiente="FINALIZAR", instruccion="", forzada=True,
                                        evaluacion=motivo_forzado)
            print(f"  [supervisor] ronda {ronda}: FINALIZAR (forzado) | {motivo_forzado}", flush=True)
            return {
                "ronda": ronda, "next_agent": "FINALIZAR", "instruccion_actual": "",
                "decisiones": [registro], "task_completed": False,
                "motivo_cierre": motivo_forzado + " Requiere revision humana.",
                "messages": [AIMessage(content=f"[ronda {ronda}] FINALIZAR (forzado)", name="supervisor")],
            }

        # --- Decision del LLM ------------------------------------------------
        contexto = (
            f"PEDIDO DE LA PERSONA:\n{pedido_original(estado)}\n\n"
            f"RONDA ACTUAL: {ronda} de {config.max_rondas_supervisor}\n\n"
            f"ESTADO:\n{_estado_resumido(estado)}\n\n"
            f"CHEQUEOS AUTOMATICOS (calculados por codigo, son confiables):\n{chequeos_a_texto(chequeos)}\n\n"
            "¿Quien interviene ahora, o es momento de FINALIZAR?"
        )

        async def decidir() -> DecisionSupervisor:
            return await asyncio.wait_for(
                decisor.ainvoke([SystemMessage(PROMPT_SUPERVISOR), HumanMessage(contexto)]),
                timeout=config.timeout_modelo,
            )

        decision = await con_reintentos(decidir)
        print(f"  [supervisor] ronda {ronda}: -> {decision.siguiente} | {decision.evaluacion}", flush=True)

        actualizacion: dict[str, Any] = {
            "ronda": ronda,
            "next_agent": decision.siguiente,
            "instruccion_actual": decision.instruccion,
            "decisiones": [RegistroDecision(ronda=ronda, **decision.model_dump())],
            "messages": [AIMessage(content=f"[ronda {ronda}] -> {decision.siguiente}. {decision.evaluacion}",
                                   name="supervisor")],
        }
        if decision.siguiente == "FINALIZAR":
            # "Confiar pero verificar": el LLM decide cerrar, el codigo decide si quedo completo.
            completa = todo_ok(chequeos)
            actualizacion["task_completed"] = completa
            actualizacion["motivo_cierre"] = (
                "Rubrica cumplida y chequeos automaticos OK." if completa else
                f"El Supervisor cerro sin completar la propuesta: {decision.evaluacion} "
                "Chequeos pendientes o fallidos: "
                + "; ".join(c.detalle for c in chequeos if c.ok is not True and c.bloqueante)
                + " Requiere revision humana."
            )
        return actualizacion

    return nodo_supervisor
