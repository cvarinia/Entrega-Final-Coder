"""Agente Investigador: averigua quien es la LSP y que senales muestra.

Responsabilidad unica: armar un PerfilLSP con fuentes. No elige servicios ni
redacta nada.

Herramientas:
    - buscar_web            (Tavily: resultados con titulo, URL y extracto)
    - leer_pagina           (Tavily: texto de una URL concreta)
    - consultar_metodologia (RAG: perfil del comprador y criterios de LSP ideal)
    - entregar_perfil       (entrega formal, validada con Pydantic)

Regla de proveniencia: toda URL citada en el perfil tiene que haber salido de
una busqueda o haber sido leida en ESTA corrida. Las herramientas se crean en
cada invocacion del nodo para poder llevar esa lista de URLs consultadas sin
mezclar ejecuciones concurrentes de distintos jobs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import correr_agente, herramienta_metodologia, ultima_entrega
from comun.config import Configuracion
from comun.recursos import Recursos
from state import Aporte, EstadoPropuesta, PerfilLSP

NOMBRE = "investigador"

PROMPT_INVESTIGADOR = """Sos el Agente Investigador de Headway, una empresa argentina que hace
localizacion de e-learning (Articulate Storyline y Rise), localizacion de Figma y DTP
multilingue PARA agencias de localizacion (LSPs). Headway no traduce: trabaja para las
agencias que traducen.

Tu unico trabajo: investigar la agencia (LSP) que indica la instruccion y entregar su perfil.
No elijas servicios ni redactes mensajes: de eso se ocupan otros agentes.

Como trabajar (tenes un presupuesto de pocas busquedas/lecturas; con 4 a 6 alcanza):
1. buscar_web con el nombre de la agencia y temas utiles: e-learning, Storyline,
   "become a vendor" / "vendor", DTP, idiomas.
2. leer_pagina del sitio oficial y, si existe, de la pagina de alta de proveedores.
3. Si dudas que buscar, consultar_metodologia ("criterios de LSP ideal").
4. entregar_perfil OBLIGATORIAMENTE al final, con UN argumento "perfil" que tenga todos
   los campos. Es mejor un perfil con 2 o 3 senales bien respaldadas que seguir buscando.
   Despues responde en una o dos lineas en espanol rioplatense.

Reglas:
- Cada senal lleva la URL donde la viste. Solo podes citar URLs que aparecieron en
  buscar_web o que leiste con leer_pagina. Nunca inventes una URL ni un dato.
- Criterios posibles de las senales: servicio_elearning, trabaja_con_articulate,
  recluta_proveedores, muchos_idiomas (mas de 15), en_crecimiento, ofrece_dtp_multimedia.
  Solo agrega una senal si la fuente la respalda.
- Canales de contacto: solo canales PUBLICOS de la empresa (portal o formulario de
  proveedores, direcciones genericas como vendors@..., pagina de LinkedIn de la empresa).
  Nunca emails, telefonos ni perfiles personales.
- Si entregar_perfil devuelve un error, lee el detalle, corregi y volve a entregar.
"""


def crear_herramientas(recursos: Recursos, max_consultas_web: int = 8) -> list:
    """Herramientas de UNA corrida del investigador (con su registro de URLs y su presupuesto)."""
    urls_consultadas: set[str] = set()
    consultas = {"hechas": 0}

    def presupuesto_agotado() -> dict[str, Any] | None:
        """Presupuesto de busquedas, aplicado por CODIGO.

        En la primera prueba real el investigador encontro todo lo necesario en
        5 o 6 consultas, pero siguio buscando hasta quedarse sin pasos y nunca
        entrego. Pedirle "se breve" en el prompt no alcanzo: el limite lo pone
        la herramienta, que deja de buscar y le dice que entregue.
        """
        if consultas["hechas"] >= max_consultas_web:
            return {"error": f"Presupuesto agotado: ya hiciste {max_consultas_web} busquedas o lecturas.",
                    "sugerencia": "Llama AHORA a entregar_perfil con lo que ya encontraste."}
        consultas["hechas"] += 1
        return None

    @tool
    async def buscar_web(consulta: str, max_resultados: int = 5) -> dict[str, Any]:
        """Busca en internet. Devuelve titulo, URL y un extracto de cada resultado.

        Args:
            consulta: Lo que buscas, por ejemplo "Acme Translations e-learning Storyline".
            max_resultados: Entre 1 y 8.
        """
        if (agotado := presupuesto_agotado()) is not None:
            return agotado
        max_resultados = max(1, min(max_resultados, 8))
        try:
            resultados = await recursos.web.buscar(consulta, max_resultados)
        except Exception as error:  # noqa: BLE001 - el agente tiene que poder seguir
            return {"error": f"Fallo la busqueda: {error}", "sugerencia": "Reintenta con otra consulta."}
        if not resultados:
            return {"error": "Sin resultados.", "sugerencia": "Usa menos palabras o el nombre exacto."}
        urls_consultadas.update(r.url for r in resultados)
        return {"resultados": [r.model_dump() for r in resultados]}

    @tool
    async def leer_pagina(url: str) -> dict[str, Any]:
        """Devuelve el texto de una pagina web (los primeros miles de caracteres).

        Args:
            url: Direccion completa, empezando con http:// o https://.
        """
        if (agotado := presupuesto_agotado()) is not None:
            return agotado
        try:
            texto = await recursos.web.leer(url)
        except Exception as error:  # noqa: BLE001
            return {"error": f"No se pudo leer {url}: {error}", "sugerencia": "Proba con otra URL de la busqueda."}
        urls_consultadas.add(url)
        return {"url": url, "texto": texto}

    @tool
    async def entregar_perfil(perfil: PerfilLSP) -> dict[str, Any]:
        """Entrega formalmente el perfil de la LSP. Llamala UNA vez, al final.

        Valida que cada URL citada (sitio, senales y canales) haya sido consultada en
        esta investigacion. Si no, devuelve error y tenes que corregir.
        """
        no_consultadas = sorted(u for u in perfil.urls_citadas() if u not in urls_consultadas)
        if no_consultadas:
            return {
                "error": f"Estas URLs no salieron de buscar_web ni de leer_pagina: {no_consultadas}.",
                "sugerencia": "Cita solo URLs que consultaste, o leelas primero con leer_pagina.",
            }
        return {"aceptado": True, "perfil": perfil.model_dump()}

    return [buscar_web, leer_pagina, herramienta_metodologia(recursos.buscador, "investigador"), entregar_perfil]


def crear_nodo_investigador(modelo: BaseChatModel, config: Configuracion, recursos: Recursos):
    async def nodo_investigador(estado: EstadoPropuesta) -> dict[str, Any]:
        instruccion = estado["instruccion_actual"]
        print(f"  [investigador] {instruccion}", flush=True)

        herramientas = crear_herramientas(recursos, config.max_consultas_web)
        agente = create_react_agent(modelo, herramientas, prompt=PROMPT_INVESTIGADOR, name=NOMBRE)
        evidencia, resumen = await correr_agente(agente, instruccion, config)

        entrega = ultima_entrega(evidencia, "entregar_perfil")
        actualizacion: dict[str, Any] = {}
        if entrega:
            perfil = PerfilLSP.model_validate(entrega["perfil"])
            actualizacion.update(perfil=perfil, ronda_perfil=estado["ronda"])
        else:
            resumen += " (ATENCION: no se registro ningun perfil con entregar_perfil)"

        aporte = Aporte(agente=NOMBRE, ronda=estado["ronda"], instruccion=instruccion,
                        resumen=resumen, entrego=bool(entrega), evidencia=evidencia)
        return {**actualizacion, "aportes": [aporte], "messages": [AIMessage(content=resumen, name=NOMBRE)]}

    return nodo_investigador
