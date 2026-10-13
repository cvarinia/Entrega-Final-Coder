"""Agente Especialista de servicios: cruza el perfil de la LSP con el catalogo.

Responsabilidad unica: elegir de 1 a 4 servicios de Headway que respondan a
las senales del perfil, y justificar cada uno. No investiga ni redacta.

Herramientas:
    - buscar_servicios      (RAG hibrido sobre el catalogo, DIVERSIFICADO por servicio)
    - ver_ficha             (ficha completa de un servicio)
    - consultar_metodologia (RAG: dolores de la LSP y como los resuelve Headway)
    - entregar_match        (entrega formal; el CODIGO completa prioridad y tarifa)

Dos defensas contra alucinaciones del catalogo:
    1. Filtro de metadatos: buscar_servicios nunca devuelve servicios
       `en_evaluacion` (el agente no puede ofrecer lo que no ve).
    2. entregar_match valida cada servicio_id contra el catalogo y COPIA del
       catalogo el nombre, la prioridad y la tarifa: el LLM solo elige que y
       por que; los datos duros no pasan por el modelo.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import correr_agente, herramienta_metodologia, ultima_entrega
from agents.reglas import frases_exageradas
from comun.config import Configuracion
from comun.recursos import Recursos
from rag.hibrido import filtro_servicios_ofrecibles
from state import Aporte, EstadoPropuesta, MatchServicios, ServicioElegido, ServicioPropuesto

NOMBRE = "especialista"

PROMPT_ESPECIALISTA = """Sos el Especialista de servicios de Headway (localizacion de e-learning y
DTP multilingue para agencias de localizacion). Recibis el perfil validado de una LSP y
elegis que servicios de Headway ofrecerle.

Como trabajar:
1. Por cada senal importante del perfil, buscar_servicios con la necesidad en una frase
   (por ejemplo "agencia que localiza cursos de Storyline a muchos idiomas").
2. Si dudas, ver_ficha del servicio o consultar_metodologia ("dolores de la LSP").
3. entregar_match OBLIGATORIAMENTE al final, con 1 a 4 servicios. Despues responde en
   una o dos lineas en espanol rioplatense.

Reglas:
- Lo principal de Headway es la LOCALIZACION de e-learning (Storyline, Rise, Figma) y el DTP
  (formateo de cursos y documentos en cualquier idioma). Si la LSP ofrece DTP o trabaja con
  documentos, el DTP multilingue es una muy buena eleccion.
- Los servicios de tipo "add_on" (creacion de cursos, voces IA) son ADICIONALES: solo junto a un
  servicio principal y nunca como el argumento central.
- Prioriza los servicios de prioridad "estrella"; los "nucleo" si una senal los justifica.
- Cada servicio elegido responde a una senal concreta del perfil: decilo en el motivo.
- Usa solo servicio_id que hayan aparecido en buscar_servicios o ver_ficha.
- Los servicios de prioridad "complemento" (QA) solo si la instruccion dice que la persona
  los pidio; en ese caso usa incluir_complementos=true.
- Si la instruccion trae un comentario de la revision humana, respetalo al pie de la letra.
- Si entregar_match devuelve un error, corregi y volve a entregar.
"""


def crear_herramientas(recursos: Recursos) -> list:
    @tool
    async def buscar_servicios(necesidad: str, incluir_complementos: bool = False) -> dict[str, Any]:
        """Busca en el catalogo de Headway los servicios que responden a una necesidad.

        Devuelve como mucho un resultado por servicio (el fragmento mas relevante), con
        su prioridad y tarifa. Nunca devuelve servicios que no se ofrecen.

        Args:
            necesidad: La necesidad de la LSP, en una frase.
            incluir_complementos: True solo si la persona pidio QA/testing.
        """
        resultados = await recursos.buscador.buscar_diversificado(
            "catalogo", necesidad, filtro_servicios_ofrecibles(incluir_complementos)
        )
        if not resultados:
            return {"error": "No hay servicios para esa necesidad.", "sugerencia": "Describi la necesidad de otra forma."}
        return {"servicios": [
            {
                "servicio_id": r.metadatos["servicio_id"],
                "servicio": r.metadatos["servicio"],
                "prioridad": r.metadatos["prioridad"],
                "tarifa": r.metadatos["tarifa"],
                "tipo": r.metadatos.get("tipo", "servicio"),
                "fragmento": r.texto.split("\n\n", 1)[-1],
            }
            for r in resultados
        ]}

    @tool
    async def ver_ficha(servicio_id: str) -> dict[str, Any]:
        """Devuelve la ficha completa de un servicio del catalogo.

        Args:
            servicio_id: Por ejemplo "localizacion_storyline".
        """
        ficha = await recursos.catalogo.ficha(servicio_id)
        if ficha is None or not ficha.ofrecible:
            return {"error": f"'{servicio_id}' no es un servicio que se ofrezca.",
                    "servicios_validos": await recursos.catalogo.ids_ofrecibles()}
        return ficha.model_dump()

    @tool
    async def entregar_match(servicios: list[ServicioElegido], argumento_central: str) -> dict[str, Any]:
        """Entrega formalmente los servicios elegidos (1 a 4). Llamala UNA vez, al final.

        Args:
            servicios: Cada uno con servicio_id y el motivo (que senal de la LSP responde).
            argumento_central: La idea principal de la propuesta, en una o dos frases.
        """
        propuestos: list[ServicioPropuesto] = []
        invalidos: list[str] = []
        for elegido in servicios:
            ficha = await recursos.catalogo.ficha(elegido.servicio_id)
            if ficha is None or not ficha.ofrecible:
                invalidos.append(elegido.servicio_id)
                continue
            if any(p.servicio_id == ficha.servicio_id for p in propuestos):
                continue  # repetido
            propuestos.append(ServicioPropuesto(
                servicio_id=ficha.servicio_id, motivo=elegido.motivo,
                servicio=ficha.servicio, prioridad=ficha.prioridad, tarifa=ficha.tarifa, tipo=ficha.tipo,
            ))
        if invalidos:
            return {"error": f"Servicios inexistentes o que no se ofrecen: {invalidos}.",
                    "servicios_validos": await recursos.catalogo.ids_ofrecibles()}
        if exageradas := frases_exageradas(argumento_central):
            return {"error": f"Tono exagerado en el argumento central: {exageradas}.",
                    "sugerencia": "Headway es un estudio pequeno, cercano y confiable: nada de escalar ni superlativos."}
        if propuestos and all(p.tipo == "add_on" for p in propuestos):
            return {"error": "Solo elegiste adicionales (add_on). Un adicional nunca es el centro de la propuesta.",
                    "sugerencia": "Inclui al menos un servicio principal: localizacion (Storyline, Rise, Figma) o DTP."}
        # Los adicionales siempre van al final: el orden comunica que es lo principal.
        propuestos.sort(key=lambda p: p.tipo == "add_on")
        try:
            match = MatchServicios(servicios=propuestos, argumento_central=argumento_central)
        except ValueError as error:
            return {"error": f"El match no cumple el contrato: {error}"}
        return {"aceptado": True, "match": match.model_dump()}

    return [buscar_servicios, ver_ficha, herramienta_metodologia(recursos.buscador, "especialista_servicios"),
            entregar_match]


def entrada_especialista(estado: EstadoPropuesta) -> str:
    """Instruccion del Supervisor + SOLO los datos validados que necesita."""
    partes = [f"INSTRUCCION DEL SUPERVISOR:\n{estado['instruccion_actual']}"]
    perfil = estado.get("perfil")
    if perfil is not None:
        partes.append("PERFIL VALIDADO DE LA LSP:\n" + json.dumps(perfil.model_dump(), ensure_ascii=False, indent=1))
    decision = estado.get("decision_match")
    if decision is not None and not decision.aprobado:
        partes.append(f"REVISION HUMANA DEL MATCH ANTERIOR: rechazado. Comentario: {decision.comentario or '(sin comentario)'}")
    return "\n\n".join(partes)


def crear_nodo_especialista(modelo: BaseChatModel, config: Configuracion, recursos: Recursos):
    async def nodo_especialista(estado: EstadoPropuesta) -> dict[str, Any]:
        print(f"  [especialista] {estado['instruccion_actual']}", flush=True)
        agente = create_react_agent(modelo, crear_herramientas(recursos), prompt=PROMPT_ESPECIALISTA, name=NOMBRE)
        evidencia, resumen = await correr_agente(agente, entrada_especialista(estado), config)

        entrega = ultima_entrega(evidencia, "entregar_match")
        actualizacion: dict[str, Any] = {}
        if entrega:
            actualizacion.update(
                match=MatchServicios.model_validate(entrega["match"]),
                ronda_match=estado["ronda"],
                version_match=estado.get("version_match", 0) + 1,  # un match nuevo necesita aprobacion nueva
                decision_match=None,  # la decision anterior era sobre OTRO match
            )
        else:
            resumen += " (ATENCION: no se registro ningun match con entregar_match)"

        aporte = Aporte(agente=NOMBRE, ronda=estado["ronda"], instruccion=estado["instruccion_actual"],
                        resumen=resumen, entrego=bool(entrega), evidencia=evidencia)
        return {**actualizacion, "aportes": [aporte], "messages": [AIMessage(content=resumen, name=NOMBRE)]}

    return nodo_especialista
