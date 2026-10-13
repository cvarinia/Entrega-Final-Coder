"""Pruebas del sistema completo SIN gastar la API (modelo y web simulados).

Lo que es real: las herramientas, sus validaciones, el RAG (Chroma de prueba
con embeddings simulados), los chequeos, el grafo y el checkpointer.
Lo simulado: las decisiones del LLM (guion) y las paginas web.

Necesitan Chroma levantado (docker compose up -d chroma); si no, se saltean.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from agents.validacion import chequeos_automaticos, todo_ok
from app.hitl import pedido_pendiente
from comun.config import Configuracion
from comun.modelo_simulado import ModeloSimulado
from comun.recursos import Recursos
from graph import checkpointer_en_memoria, config_de_ejecucion, construir_grafo, estado_inicial, rutear
from rag.almacen import AlmacenChroma
from rag.catalogo import Catalogo
from rag.embeddings import EmbeddingsSimulados
from rag.hibrido import BuscadorHibrido
from rag.indexar import asegurar_indice
from state import DecisionSupervisor
from tests import guiones as g

CONFIG = Configuracion.para_pruebas()
PEDIDO = "Quiero ofrecerle localizacion de e-learning a Acme Localization."


def decision(siguiente: str, instruccion: str = "Hacelo.") -> DecisionSupervisor:
    return DecisionSupervisor(siguiente=siguiente, instruccion=instruccion, evaluacion="(guion)")


@pytest.fixture
async def recursos() -> Recursos:
    try:
        almacen = await AlmacenChroma.conectar(CONFIG)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Chroma no esta disponible: {error}")
    vectorizador = EmbeddingsSimulados()
    await asegurar_indice(CONFIG, almacen, vectorizador)
    return Recursos(buscador=BuscadorHibrido(almacen, vectorizador), catalogo=Catalogo(almacen), web=g.web_simulada())


async def arrancar(recursos: Recursos, modelo: ModeloSimulado, hilo: str, checkpointer=None, pedido: str = PEDIDO):
    grafo = construir_grafo(CONFIG, recursos, modelo=modelo, checkpointer=checkpointer or checkpointer_en_memoria())
    cfg = config_de_ejecucion(hilo, CONFIG)
    return grafo, cfg, await grafo.ainvoke(estado_inicial(pedido), cfg)


def guion_feliz(redaccion=None) -> ModeloSimulado:
    return ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad(), *(redaccion or g.redaccion())],
        decisiones=[decision("investigador"), decision("especialista"), decision("redactor"), decision("FINALIZAR")],
    )


# --------------------------------------------------------------------------


async def test_flujo_completo_con_aprobacion(recursos):
    grafo, cfg, resultado = await arrancar(recursos, guion_feliz(), "feliz")

    # 1) Se pauso ANTES del redactor, con el match para revisar.
    pedido = pedido_pendiente(resultado)
    assert pedido is not None and pedido["accion"] == "aprobar_servicios"
    assert [s["servicio_id"] for s in pedido["servicios"]] == ["localizacion_storyline", "dtp_multilingue"]
    assert (await grafo.aget_state(cfg)).next == ("aprobacion_match",)
    assert resultado.get("propuesta") is None

    # 2) Se aprueba y el grafo sigue hasta el final.
    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "carla"}), cfg)
    assert final["task_completed"] is True, final["motivo_cierre"]
    assert final["perfil"].puntaje == 9 and final["perfil"].prioridad_contacto == "alta"
    # La tarifa la completo el CODIGO desde el catalogo, no el LLM.
    assert {s.tarifa for s in final["match"].servicios} == {"14 USD/h"}
    respuesta = final["messages"][-1]
    assert respuesta.name == "sintesis" and g.MENSAJE_OK in respuesta.content
    assert [d.siguiente for d in final["decisiones"]] == ["investigador", "especialista", "redactor", "FINALIZAR"]


async def test_rechazo_vuelve_al_especialista_y_pide_aprobacion_de_nuevo(recursos):
    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad(),
                    *g.especialidad([("localizacion_storyline", "Sus cursos de Storyline en 40 idiomas.")]),
                    *g.redaccion()],
        decisiones=[decision("investigador"), decision("especialista"), decision("redactor"),
                    decision("especialista", "Saca DTP: la persona solo quiere ofrecer e-learning."),
                    decision("redactor"), decision("FINALIZAR")],
    )
    grafo, cfg, resultado = await arrancar(recursos, modelo, "rechazo")
    assert pedido_pendiente(resultado)["version_match"] == 1

    # Rechazo -> supervisor -> especialista (match v2) -> supervisor -> NUEVA pausa
    segundo = await grafo.ainvoke(Command(resume={"aprobado": False, "revisor": "carla",
                                                   "comentario": "Solo e-learning, sin DTP"}), cfg)
    pedido = pedido_pendiente(segundo)
    assert pedido is not None and pedido["version_match"] == 2
    assert [s["servicio_id"] for s in pedido["servicios"]] == ["localizacion_storyline"]

    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "carla"}), cfg)
    assert final["task_completed"] is True
    assert final["version_aprobada"] == final["version_match"] == final["version_match_propuesta"] == 2


async def test_precio_inventado_se_rechaza_y_el_redactor_corrige(recursos):
    inventado = g.MENSAJE_OK.replace("14 USD/h", "15 USD/h")
    grafo, cfg, _ = await arrancar(recursos, guion_feliz(g.redaccion([inventado, g.MENSAJE_OK])), "precio")
    final = await grafo.ainvoke(Command(resume={"aprobado": True}), cfg)

    entregas = [ev for a in final["aportes"] if a.agente == "redactor" for ev in a.evidencia
                if ev.herramienta == "entregar_propuesta"]
    assert entregas[0].error is not None and "15" in str(entregas[0].resultado["problemas"])
    assert entregas[1].error is None
    assert "15 USD" not in final["propuesta"].mensaje and final["task_completed"] is True


async def test_fuente_no_consultada_se_rechaza(recursos):
    inventada = {**g.PERFIL, "senales": [{"criterio": "en_crecimiento", "descripcion": "Abrio oficina nueva en Madrid.",
                                           "fuente_url": "https://noticias.example/acme"}]}
    modelo = ModeloSimulado(
        respuestas=[g.llamada("buscar_web", consulta="Acme"), g.llamada("entregar_perfil", perfil=inventada),
                    g.llamada("entregar_perfil", perfil=g.PERFIL), g.final("Listo.")],
        decisiones=[decision("investigador"), decision("FINALIZAR")],
    )
    _, _, final = await arrancar(recursos, modelo, "fuente")
    entregas = [ev for ev in final["aportes"][0].evidencia if ev.herramienta == "entregar_perfil"]
    assert entregas[0].error is not None and "noticias.example" in entregas[0].error
    assert final["perfil"].nombre == "Acme Localization"
    # Cerro sin match ni propuesta: el codigo NO la da por completa aunque el LLM haya dicho FINALIZAR.
    assert final["task_completed"] is False


async def test_servicio_en_evaluacion_no_se_puede_entregar(recursos):
    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(),
                    g.llamada("entregar_match", servicios=[{"servicio_id": "avatares_ia", "motivo": "Les gustaria."}],
                              argumento_central="Avatares para sus cursos en muchos idiomas."),
                    g.final("No pude.")],
        decisiones=[decision("investigador"), decision("especialista"), decision("FINALIZAR")],
    )
    _, _, final = await arrancar(recursos, modelo, "avatares")
    entrega = [ev for ev in final["aportes"][1].evidencia if ev.herramienta == "entregar_match"][0]
    assert entrega.error is not None and "avatares_ia" in entrega.error
    assert final["match"] is None


async def test_qa_no_pedido_falla_el_chequeo(recursos):
    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad([("qa_formato", "Pueden necesitar control de formato.")])],
        decisiones=[decision("investigador"), decision("especialista"), decision("FINALIZAR")],
    )
    _, _, final = await arrancar(recursos, modelo, "qa")
    chequeos = {c.nombre: c for c in await chequeos_automaticos(final, recursos.catalogo)}
    assert chequeos["complementos_pedidos"].ok is False
    assert not todo_ok(list(chequeos.values()))

    # Si la persona SI pidio QA, el mismo match pasa ese chequeo.
    _, _, final2 = await arrancar(recursos, ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad([("qa_formato", "Pidieron QA.")])],
        decisiones=[decision("investigador"), decision("especialista"), decision("FINALIZAR")],
    ), "qa-pedido", pedido="Ofrecele QA de formato a Acme Localization.")
    chequeos2 = {c.nombre: c for c in await chequeos_automaticos(final2, recursos.catalogo)}
    assert chequeos2["complementos_pedidos"].ok is True


async def test_limite_de_rondas_cierra_por_codigo(recursos):
    config = Configuracion.para_pruebas(max_rondas_supervisor=1)
    modelo = ModeloSimulado(respuestas=[*g.investigacion()], decisiones=[decision("investigador")])
    grafo = construir_grafo(config, recursos, modelo=modelo, checkpointer=checkpointer_en_memoria())
    final = await grafo.ainvoke(estado_inicial(PEDIDO), config_de_ejecucion("limite", config))
    assert final["decisiones"][-1].forzada is True and final["task_completed"] is False
    assert "NO quedó completa" in final["messages"][-1].content


async def test_respuesta_humana_invalida(recursos):
    grafo, cfg, _ = await arrancar(recursos, guion_feliz(), "invalida")
    with pytest.raises(ValidationError):
        await grafo.ainvoke(Command(resume={"aprobado": "tal vez"}), cfg)


async def test_serializador_redis_reanuda_con_modelos(recursos):
    """Regresion de la PE7: con el serializador JSON de Redis, el estado tiene que volver como modelos."""
    from langgraph.checkpoint.memory import InMemorySaver

    from graph import serializador_redis
    from state import MatchServicios, PerfilLSP

    grafo, cfg, resultado = await arrancar(recursos, guion_feliz(), "serde",
                                           checkpointer=InMemorySaver(serde=serializador_redis()))
    assert pedido_pendiente(resultado) is not None
    valores = (await grafo.aget_state(cfg)).values
    assert isinstance(valores["perfil"], PerfilLSP) and isinstance(valores["match"], MatchServicios)
    final = await grafo.ainvoke(Command(resume={"aprobado": True}), cfg)
    assert final["task_completed"] is True


def test_rutear_no_deja_redactar_sin_aprobacion():
    from state import MatchServicios, ServicioPropuesto

    match = MatchServicios(servicios=[ServicioPropuesto(servicio_id="x_y_z", motivo="motivo suficiente",
                                                        servicio="X", prioridad="estrella", tarifa="14 USD/h")],
                           argumento_central="Argumento central suficientemente largo.")
    base = {"next_agent": "redactor", "match": match, "version_match": 1}
    assert rutear({**base, "version_aprobada": 0}) == "aprobacion_match"
    assert rutear({**base, "version_aprobada": 1}) == "redactor"
    assert rutear({**base, "next_agent": "FINALIZAR"}) == "sintesis"


async def test_agente_estancado_cierra_sin_gastar_rondas(recursos):
    """Regresion de la primera prueba real: pedido sin nombre de agencia.

    El investigador no puede entregar dos veces seguidas -> el codigo cierra
    en la ronda 3 en vez de insistir hasta el limite de rondas.
    """
    modelo = ModeloSimulado(
        respuestas=[g.final("Falta el nombre de la agencia."), g.final("Sigue faltando el nombre.")],
        decisiones=[decision("investigador"), decision("investigador")],
    )
    _, _, final = await arrancar(recursos, modelo, "estancado", pedido="Quiero ofrecerle e-learning a la agencia <nombre>")
    assert [d.siguiente for d in final["decisiones"]] == ["investigador", "investigador", "FINALIZAR"]
    assert final["decisiones"][-1].forzada is True and final["task_completed"] is False
    assert "Sigue faltando el nombre" in final["motivo_cierre"]


async def test_presupuesto_de_busquedas_del_investigador(recursos):
    """Regresion de la prueba con Vistatec: el investigador buscaba sin parar y nunca entregaba."""
    config = Configuracion.para_pruebas(max_consultas_web=2)
    modelo = ModeloSimulado(
        respuestas=[g.llamada("buscar_web", consulta="a"), g.llamada("leer_pagina", url=g.VENDORS),
                    g.llamada("buscar_web", consulta="otra mas"),   # tercera: ya no hay presupuesto
                    g.llamada("entregar_perfil", perfil=g.PERFIL), g.final("Listo.")],
        decisiones=[decision("investigador"), decision("FINALIZAR")],
    )
    grafo = construir_grafo(config, recursos, modelo=modelo, checkpointer=checkpointer_en_memoria())
    final = await grafo.ainvoke(estado_inicial(PEDIDO), config_de_ejecucion("presupuesto", config))
    tercera = final["aportes"][0].evidencia[2]
    assert tercera.error is not None and "Presupuesto agotado" in tercera.error
    assert final["perfil"] is not None  # igual entrego con lo que tenia


def test_limpiar_pagina_saca_menus():
    from comun.web import limpiar_pagina

    texto = "Company\nServices\nAI Hub\nWe localize e-learning courses into more than 60 languages for global brands.\nAI Hub"
    assert limpiar_pagina(texto) == "We localize e-learning courses into more than 60 languages for global brands."


async def test_adicional_solo_no_puede_ser_el_match(recursos):
    """Revision de Carla: crear cursos desde cero es un adicional, nunca el centro."""
    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(),
                    *g.especialidad([("creacion_cursos", "Ofrecen desarrollo de contenidos."),
                                     ("localizacion_storyline", "Cursos de Storyline en 40 idiomas.")])],
        decisiones=[decision("investigador"), decision("especialista"), decision("FINALIZAR")],
    )
    _, _, final = await arrancar(recursos, modelo, "adicional")
    ids = [s.servicio_id for s in final["match"].servicios]
    assert ids == ["localizacion_storyline", "creacion_cursos"]  # el adicional quedo al final

    solo_adicional = ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad([("creacion_cursos", "Ofrecen desarrollo de contenidos.")])],
        decisiones=[decision("investigador"), decision("especialista"), decision("FINALIZAR")],
    )
    _, _, final2 = await arrancar(recursos, solo_adicional, "solo-adicional")
    assert final2["match"] is None
