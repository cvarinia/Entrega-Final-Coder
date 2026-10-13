"""Chequeos automaticos del trabajo de los agentes (codigo, no LLM).

Mismo patron de la PE6/PE7: lo que se puede verificar con certeza se verifica
con codigo, y el resultado se le pasa al Supervisor como parte de su prompt.
El LLM queda para lo que no se puede codificar (por ejemplo, juzgar si un
servicio responde de verdad a una senal).

Chequeos, en el orden del flujo:
    1. perfil_entregado
    2. fuentes_consultadas     toda URL citada aparece en la evidencia del investigador
    3. canal_de_contacto       (advertencia, no bloquea)
    4. match_vigente           el match es posterior al ultimo perfil
    5. servicios_ofrecibles    cada servicio existe en el catalogo y esta activo
    6. complementos_pedidos    QA solo si la persona lo pidio
    7. match_aprobado          una persona aprobo ESTA version del match
    8. propuesta_vigente       la propuesta se escribio sobre el match aprobado
    9. reglas_de_la_propuesta  largo, asunto, prueba y precios del catalogo
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from agents.reglas import pide_complementos, revisar_propuesta
from comun.mensajes import texto_de_mensaje
from rag.catalogo import Catalogo
from state import Aporte, EstadoPropuesta


@dataclass(frozen=True)
class Chequeo:
    nombre: str
    ok: bool | None   # None = pendiente / no aplica todavia
    detalle: str
    bloqueante: bool = True


def pedido_original(estado: EstadoPropuesta) -> str:
    for mensaje in estado.get("messages", []):
        if isinstance(mensaje, HumanMessage):
            return texto_de_mensaje(mensaje)
    return ""


def _urls_en_evidencia(aportes: list[Aporte]) -> set[str]:
    """URLs que el investigador realmente vio: resultados de busqueda y paginas leidas."""
    urls: set[str] = set()
    for aporte in aportes:
        if aporte.agente != "investigador":
            continue
        for ev in aporte.evidencia:
            if ev.error is not None or not isinstance(ev.resultado, dict):
                continue
            if ev.herramienta == "buscar_web":
                urls |= {r.get("url", "") for r in ev.resultado.get("resultados", [])}
            elif ev.herramienta == "leer_pagina":
                urls.add(ev.resultado.get("url", ""))
    return urls


async def chequeos_automaticos(estado: EstadoPropuesta, catalogo: Catalogo) -> list[Chequeo]:
    chequeos: list[Chequeo] = []
    perfil, match, propuesta = estado.get("perfil"), estado.get("match"), estado.get("propuesta")

    # 1-3. Perfil -------------------------------------------------------------
    if perfil is None:
        chequeos.append(Chequeo("perfil_entregado", False, "El investigador todavia no entrego un perfil."))
        return chequeos
    chequeos.append(Chequeo("perfil_entregado", True,
                            f"Perfil de {perfil.nombre}: puntaje {perfil.puntaje}/11 (prioridad {perfil.prioridad_contacto})."))

    no_vistas = sorted(perfil.urls_citadas() - _urls_en_evidencia(estado.get("aportes", [])))
    chequeos.append(Chequeo(
        "fuentes_consultadas", not no_vistas,
        "Todas las URLs citadas fueron consultadas." if not no_vistas else
        f"URLs citadas que no aparecen en la evidencia del investigador: {no_vistas}.",
    ))
    chequeos.append(Chequeo(
        "canal_de_contacto", True if perfil.canales_contacto else None,
        f"{len(perfil.canales_contacto)} canal(es) publicos." if perfil.canales_contacto else
        "El perfil no tiene canales de contacto: el redactor no sabra por donde escribir.",
        bloqueante=False,
    ))

    # 4-6. Match ---------------------------------------------------------------
    if match is None or estado.get("ronda_match", 0) <= estado.get("ronda_perfil", 0):
        chequeos.append(Chequeo("match_vigente", False, "No hay un match de servicios posterior al ultimo perfil."))
        return chequeos
    chequeos.append(Chequeo("match_vigente", True,
                            f"Match v{estado.get('version_match', 0)}: "
                            + ", ".join(s.servicio_id for s in match.servicios) + "."))

    no_ofrecibles = []
    for servicio in match.servicios:
        ficha = await catalogo.ficha(servicio.servicio_id)
        if ficha is None or not ficha.ofrecible or ficha.tarifa != servicio.tarifa:
            no_ofrecibles.append(servicio.servicio_id)
    chequeos.append(Chequeo(
        "servicios_ofrecibles", not no_ofrecibles,
        "Todos los servicios existen, estan activos y con la tarifa del catalogo." if not no_ofrecibles else
        f"Servicios que no se ofrecen o con tarifa distinta al catalogo: {no_ofrecibles}.",
    ))

    complementos = [s.servicio_id for s in match.servicios if s.prioridad == "complemento"]
    permitido = pide_complementos(pedido_original(estado))
    chequeos.append(Chequeo(
        "complementos_pedidos", not complementos or permitido,
        "Sin servicios complementarios no pedidos." if not complementos or permitido else
        f"El match incluye {complementos} (complemento) y la persona no lo pidio: hay que sacarlo.",
    ))

    principal_primero = match.servicios[0].tipo != "add_on"
    chequeos.append(Chequeo(
        "adicionales_como_adicionales", principal_primero,
        "El servicio principal va primero; los adicionales, al final." if principal_primero else
        "El match empieza por un adicional (add_on): el centro tiene que ser localizacion o DTP.",
    ))

    # 7. Aprobacion humana -------------------------------------------------------
    decision = estado.get("decision_match")
    aprobado = estado.get("version_aprobada", 0) == estado.get("version_match", 0)
    if not aprobado:
        if decision is not None and not decision.aprobado:
            chequeos.append(Chequeo("match_aprobado", False,
                                    f"La persona RECHAZO el match. Comentario: {decision.comentario or '(sin comentario)'}"))
        else:
            chequeos.append(Chequeo("match_aprobado", None,
                                    "Pendiente: la aprobacion humana se pide automaticamente al pasar al redactor."))
        return chequeos
    chequeos.append(Chequeo("match_aprobado", True,
                            f"Aprobado por {decision.revisor if decision else '?'}."))

    # 8-9. Propuesta ------------------------------------------------------------
    if propuesta is None or estado.get("version_match_propuesta") != estado.get("version_match"):
        chequeos.append(Chequeo("propuesta_vigente", False, "No hay una propuesta escrita sobre el match aprobado."))
        return chequeos
    chequeos.append(Chequeo("propuesta_vigente", True, f"Propuesta por {propuesta.canal}, en {propuesta.idioma}."))

    problemas = revisar_propuesta(propuesta, match)
    chequeos.append(Chequeo(
        "reglas_de_la_propuesta", not problemas,
        "Cumple largo, asunto, oferta de prueba y precios del catalogo." if not problemas else " ".join(problemas),
    ))
    return chequeos


def todo_ok(chequeos: list[Chequeo]) -> bool:
    """True si todos los chequeos bloqueantes dieron OK y ninguno fallo."""
    return all(c.ok is True for c in chequeos if c.bloqueante) and not any(c.ok is False for c in chequeos) \
        and any(c.nombre == "reglas_de_la_propuesta" for c in chequeos)


def chequeos_a_texto(chequeos: list[Chequeo]) -> str:
    marca = {True: "OK", False: "FALLA", None: "PENDIENTE"}
    return "\n".join(f"- [{marca[c.ok]}] {c.nombre}: {c.detalle}" for c in chequeos)
