"""Agente Estratega/Redactor: elige el canal y escribe el primer contacto.

Responsabilidad unica: con el perfil y el match APROBADO por una persona,
elegir canal y piezas y redactar el primer mensaje. No investiga ni cambia
los servicios.

Herramientas:
    - consultar_metodologia (RAG: reglas de contacto, argumentos de valor, reglas de precio)
    - entregar_propuesta    (entrega formal: valida largo, asunto, prueba y PRECIOS)

La herramienta de entrega aplica las reglas de agents/reglas.py. Si el
mensaje menciona un precio que no es tarifa de un servicio aprobado, la
entrega se rechaza y el agente tiene que corregir: el LLM no puede inventar
precios aunque quiera.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import correr_agente, herramienta_metodologia, ultima_entrega
from agents.reglas import revisar_propuesta
from comun.config import Configuracion
from comun.recursos import Recursos
from state import Aporte, EstadoPropuesta, MatchServicios, Propuesta

NOMBRE = "redactor"

PROMPT_REDACTOR = """Sos el Estratega y Redactor comercial de Headway (localizacion de e-learning
y DTP multilingue para agencias de localizacion). Recibis el perfil de una LSP y los
servicios APROBADOS por una persona, y armas el primer contacto.

Como trabajar:
1. consultar_metodologia para repasar: reglas de contacto (canal, largo, asunto,
   estructura), argumentos de valor y reglas de precio. Dos o tres consultas alcanzan.
2. Elegi el canal segun los canales del perfil: portal_proveedores si existe, si no
   email (direccion publica), si no linkedin.
3. Redacta el mensaje en el idioma de la LSP (si no esta claro, en ingles).
4. entregar_propuesta OBLIGATORIAMENTE. Despues responde en una linea en espanol rioplatense.

Reglas que se verifican automaticamente (si no se cumplen, la entrega se rechaza):
- Email: mensaje de 50 a 125 palabras y asunto de 4 a 7 palabras.
  Portal: 50 a 200 palabras. LinkedIn: hasta 300 caracteres.
- Estructura del email: contexto (un dato concreto del perfil) + valor + UNA pregunta final.
- Siempre ofrecer una prueba (test) e incluir "oferta_prueba" en las piezas.
- Precios: solo las tarifas de los servicios aprobados, tal cual. Si un servicio dice
  "pedir cotización", no escribas un numero. Nunca descuentos ni comparaciones de precio.
- Sin frases exageradas: nada de "exactly what you need", "scale", "leading", "the best",
  "perfect", "world-class". Se rechazan automaticamente.
- No nombrar clientes ni proyectos anteriores. Trayectoria siempre como "mas de".
- Sin llamadas telefonicas.

Posicionamiento (es lo que hace que los PMs vuelvan a elegir a Headway):
- Headway es un estudio PEQUENO, CERCANO y CONFIABLE, no una empresa para escalar a lo grande.
  Habla como un proveedor que le simplifica el trabajo al PM: un solo interlocutor, consultas
  agrupadas, entregas en tiempo y forma.
- El centro del mensaje es la localizacion de e-learning y el DTP. Los servicios marcados como
  "add_on" se mencionan al final, como algo que tambien pueden hacer.
- Cierre de referencia, aprobado por Headway (podes adaptarlo): "With over 10 years in e-learning
  localization, we work as a single point of contact for your team, reducing coordination
  overhead. Would a short test project in Storyline or Rise help demonstrate our fit?"
"""


def crear_herramientas(recursos: Recursos, match: MatchServicios | None) -> list:
    @tool
    async def entregar_propuesta(propuesta: Propuesta) -> dict[str, Any]:
        """Entrega formalmente la propuesta. Llamala al final; si devuelve error, corregi y volve a llamarla."""
        problemas = revisar_propuesta(propuesta, match)
        if problemas:
            return {"error": "La propuesta no cumple las reglas.", "problemas": problemas}
        return {"aceptada": True, "propuesta": propuesta.model_dump()}

    return [herramienta_metodologia(recursos.buscador, "estratega_redactor"), entregar_propuesta]


def entrada_redactor(estado: EstadoPropuesta) -> str:
    perfil, match = estado.get("perfil"), estado.get("match")
    return "\n\n".join([
        f"INSTRUCCION DEL SUPERVISOR:\n{estado['instruccion_actual']}",
        "PERFIL VALIDADO DE LA LSP:\n" + json.dumps(perfil.model_dump() if perfil else None, ensure_ascii=False, indent=1),
        "SERVICIOS APROBADOS POR LA PERSONA (con su tarifa del catalogo):\n"
        + json.dumps(match.model_dump() if match else None, ensure_ascii=False, indent=1),
    ])


def crear_nodo_redactor(modelo: BaseChatModel, config: Configuracion, recursos: Recursos):
    async def nodo_redactor(estado: EstadoPropuesta) -> dict[str, Any]:
        instruccion = estado["instruccion_actual"]
        match = estado.get("match")
        aprobado = match is not None and estado.get("version_aprobada", 0) == estado.get("version_match", 0)

        if not aprobado:
            # Defensa: sin match aprobado no se redacta (no se gasta ni una llamada al LLM).
            resumen = "No hay un match de servicios aprobado: no se puede redactar."
            aporte = Aporte(agente=NOMBRE, ronda=estado["ronda"], instruccion=instruccion,
                            resumen=resumen, entrego=False)
            return {"aportes": [aporte], "messages": [AIMessage(content=resumen, name=NOMBRE)]}

        print(f"  [redactor] {instruccion}", flush=True)
        agente = create_react_agent(modelo, crear_herramientas(recursos, match), prompt=PROMPT_REDACTOR, name=NOMBRE)
        evidencia, resumen = await correr_agente(agente, entrada_redactor(estado), config)

        entrega = ultima_entrega(evidencia, "entregar_propuesta")
        actualizacion: dict[str, Any] = {}
        if entrega:
            actualizacion.update(
                propuesta=Propuesta.model_validate(entrega["propuesta"]),
                version_match_propuesta=estado["version_match"],
            )
        else:
            resumen += " (ATENCION: no se registro ninguna propuesta con entregar_propuesta)"

        aporte = Aporte(agente=NOMBRE, ronda=estado["ronda"], instruccion=instruccion,
                        resumen=resumen, entrego=bool(entrega), evidencia=evidencia)
        return {**actualizacion, "aportes": [aporte], "messages": [AIMessage(content=resumen, name=NOMBRE)]}

    return nodo_redactor
