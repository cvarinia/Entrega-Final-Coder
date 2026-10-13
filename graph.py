"""Grafo principal: Supervisor + 3 agentes + aprobacion humana.

    START -> supervisor
    supervisor --(rutear)--> investigador | especialista | aprobacion_match | redactor | sintesis
    investigador     -> supervisor
    especialista     -> supervisor
    aprobacion_match --(rutear_tras_aprobacion)--> redactor (aprobado) | supervisor (rechazado)
    redactor         -> supervisor
    sintesis         -> END

Los agentes nunca se hablan entre si: todo pasa por el Supervisor. La unica
arista que no pasa por el es aprobacion -> redactor, porque la decision ya la
tomo una persona.

El grafo se compila con un CHECKPOINTER (Redis en la API, memoria en la CLI y
los tests). Cada ejecucion usa su propio thread_id (en la API, el job_id): es
lo que permite pausar en la aprobacion, reanudar despues, y que una
conversacion sobreviva a un reinicio del servidor.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.especialista import crear_nodo_especialista
from agents.investigador import crear_nodo_investigador
from agents.redactor import crear_nodo_redactor
from agents.sintesis import crear_nodo_sintesis
from agents.supervisor import crear_nodo_supervisor
from app.hitl import crear_nodo_aprobacion, rutear_tras_aprobacion
from comun.config import Configuracion
from comun.llm import construir_llm
from comun.recursos import Recursos
from state import TIPOS_DEL_ESTADO, EstadoPropuesta


def rutear(estado: EstadoPropuesta) -> str:
    """Arista condicional del Supervisor.

    La regla importante esta aca, en codigo: para llegar al redactor, la
    version actual del match tiene que estar aprobada por una persona. Si no,
    se pasa primero por la aprobacion. El LLM no puede saltearse ese paso.
    """
    destino = estado["next_agent"]
    if destino == "FINALIZAR":
        return "sintesis"
    if destino == "redactor" and estado.get("match") is not None \
            and estado.get("version_aprobada", 0) != estado.get("version_match", 0):
        return "aprobacion_match"
    return destino


def construir_grafo(
    config: Configuracion,
    recursos: Recursos,
    modelo: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo. `modelo` y `recursos` se pueden simular en los tests."""
    modelo = modelo or construir_llm(config)

    g = StateGraph(EstadoPropuesta)
    g.add_node("supervisor", crear_nodo_supervisor(modelo, config, recursos))
    g.add_node("investigador", crear_nodo_investigador(modelo, config, recursos))
    g.add_node("especialista", crear_nodo_especialista(modelo, config, recursos))
    g.add_node("aprobacion_match", crear_nodo_aprobacion())
    g.add_node("redactor", crear_nodo_redactor(modelo, config, recursos))
    g.add_node("sintesis", crear_nodo_sintesis())

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", rutear, ["investigador", "especialista", "aprobacion_match", "redactor", "sintesis"]
    )
    g.add_edge("investigador", "supervisor")
    g.add_edge("especialista", "supervisor")
    g.add_conditional_edges("aprobacion_match", rutear_tras_aprobacion, ["redactor", "supervisor"])
    g.add_edge("redactor", "supervisor")
    g.add_edge("sintesis", END)

    return g.compile(checkpointer=checkpointer)


def estado_inicial(pedido: str) -> dict[str, Any]:
    """Estado de arranque con todas las parcelas inicializadas."""
    return {
        "messages": [HumanMessage(content=pedido)],
        "ronda": 0, "instruccion_actual": "", "decisiones": [], "aportes": [],
        "task_completed": False, "motivo_cierre": "",
        "perfil": None, "ronda_perfil": 0,
        "match": None, "ronda_match": 0, "version_match": 0,
        "propuesta": None, "version_match_propuesta": 0,
        "decision_match": None, "version_aprobada": 0,
    }


def config_de_ejecucion(thread_id: str, config: Configuracion) -> dict[str, Any]:
    """Config de LangGraph para una ejecucion: su hilo de checkpoints + el techo de pasos."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.limite_recursion_grafo,
        "run_name": "propuesta_headway",  # nombre del span raiz en Phoenix
    }


# --------------------------------------------------------------------------
# Checkpointers (reutilizados de la PE7)
# --------------------------------------------------------------------------


@asynccontextmanager
async def checkpointer_redis(redis_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """Checkpointer en Redis 8: el estado sobrevive a reinicios del worker."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        saver.serde = serializador_redis()
        await saver.asetup()
        yield saver


def serializador_redis():
    """Redis guarda los checkpoints como JSON: hay que permitir los tipos del estado
    en las DOS listas, o los modelos vuelven como diccionarios al reanudar (bug de la PE7)."""
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    return JsonPlusRedisSerializer(
        allowed_json_modules=TIPOS_DEL_ESTADO,
        allowed_msgpack_modules=TIPOS_DEL_ESTADO,
    )


def checkpointer_en_memoria() -> BaseCheckpointSaver:
    """Para la CLI y los tests: misma logica de pausa/reanudacion, sin Redis."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_DEL_ESTADO))
