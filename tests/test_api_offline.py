"""Pruebas de la API + cola + worker SIN gastar la API del LLM.

Lo real: FastAPI, los contratos Pydantic de entrada, el registro y la cola de
jobs, el worker con su semaforo y la pausa/reanudacion del grafo.
Lo simulado: Redis (fakeredis, en memoria), el LLM (guion), la web; el
checkpointer es el de memoria. Chroma tiene que estar levantado (si no, se saltean).
"""

from __future__ import annotations

import asyncio
import time

import fakeredis
import httpx
import pytest

from app.jobs import RepositorioJobs
from app.main import app
from app.worker import bucle_worker
from comun.config import Configuracion
from comun.modelo_simulado import ModeloSimulado
from graph import checkpointer_en_memoria, construir_grafo
from state import DecisionSupervisor
from tests import guiones as g
from tests.test_grafo_offline import recursos  # noqa: F401 - fixture compartida

CONFIG = Configuracion.para_pruebas()


def decision(siguiente: str) -> DecisionSupervisor:
    return DecisionSupervisor(siguiente=siguiente, instruccion="Hacelo.", evaluacion="(guion)")


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str], limite: float = 20) -> dict:
    fin = time.monotonic() + limite
    while time.monotonic() < fin:
        job = (await cliente.get(f"/propuestas/{job_id}")).json()
        if job["estado"] in estados:
            return job
        await asyncio.sleep(0.1)
    raise AssertionError(f"El job no llego a {estados}: {job['estado']}")


class Entorno:
    def __init__(self, grafo) -> None:
        self.grafo = grafo

    async def __aenter__(self) -> httpx.AsyncClient:
        self.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        jobs = RepositorioJobs(self.redis)
        app.state.redis, app.state.jobs = self.redis, jobs  # lo que haria el lifespan
        self.detener = asyncio.Event()
        self.worker = asyncio.create_task(bucle_worker(self.grafo, jobs, CONFIG, self.detener))
        self.cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        return self.cliente

    async def __aexit__(self, *exc) -> None:
        self.detener.set()
        await self.worker
        await self.cliente.aclose()
        await self.redis.aclose()


def grafo_feliz(recursos):
    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad(), *g.redaccion()],
        decisiones=[decision("investigador"), decision("especialista"), decision("redactor"), decision("FINALIZAR")],
    )
    return construir_grafo(CONFIG, recursos, modelo=modelo, checkpointer=checkpointer_en_memoria())


async def test_flujo_completo_por_la_api(recursos):  # noqa: F811
    async with Entorno(grafo_feliz(recursos)) as cliente:
        r = await cliente.post("/propuestas", json={"agencia": "Acme Localization", "etiqueta": "test"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        job = await esperar(cliente, job_id, {"AWAITING_APPROVAL"})
        assert job["pedido_aprobacion"]["accion"] == "aprobar_servicios"
        assert job["agencia"] == "Acme Localization"

        r = await cliente.post(f"/propuestas/{job_id}/aprobar", json={"aprobado": True, "revisor": "carla"})
        assert r.status_code == 202
        # Responder dos veces la misma version -> 409
        assert (await cliente.post(f"/propuestas/{job_id}/aprobar", json={"aprobado": True})).status_code == 409

        job = await esperar(cliente, job_id, {"DONE", "FAILED"})
        assert job["estado"] == "DONE", job.get("error")
        assert job["resultado"]["task_completed"] is True
        assert job["resultado"]["propuesta"]["mensaje"] == g.MENSAJE_OK


@pytest.mark.parametrize("cuerpo", [
    {},                                         # sin agencia
    {"agencia": "<nombre>"},                    # el marcador de la primera prueba real
    {"agencia": "Acme", "sitio_web": "acme"},  # sitio sin http
])
async def test_entradas_invalidas_dan_422(cuerpo):
    cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    async with cliente:
        assert (await cliente.post("/propuestas", json=cuerpo)).status_code == 422


async def test_aprobacion_invalida_y_job_inexistente(recursos):  # noqa: F811
    async with Entorno(grafo_feliz(recursos)) as cliente:
        assert (await cliente.get("/propuestas/no-existe")).status_code == 404
        job_id = (await cliente.post("/propuestas", json={"agencia": "Acme"})).json()["job_id"]
        await esperar(cliente, job_id, {"AWAITING_APPROVAL"})
        assert (await cliente.post(f"/propuestas/{job_id}/aprobar", json={"aprobado": "tal vez"})).status_code == 422


async def test_script_de_pruebas_con_rechazo_y_pedido_invalido(recursos):  # noqa: F811
    """El script de las 5 pruebas oficiales, contra la API con el grafo simulado."""
    from scripts.pruebas import caso_invalido, correr_caso

    modelo = ModeloSimulado(
        respuestas=[*g.investigacion(), *g.especialidad(),
                    *g.especialidad([("localizacion_storyline", "Sus cursos de Storyline en 40 idiomas.")]),
                    *g.redaccion()],
        decisiones=[decision("investigador"), decision("especialista"), decision("redactor"),
                    decision("especialista"), decision("redactor"), decision("FINALIZAR")],
    )
    grafo = construir_grafo(CONFIG, recursos, modelo=modelo, checkpointer=checkpointer_en_memoria())
    async with Entorno(grafo) as cliente:
        assert (await caso_invalido(cliente))["ok"] is True
        caso = {"agencia": "Acme Localization", "aprobacion": "rechazar_y_aprobar", "comentario": "Solo e-learning."}
        r = await correr_caso(cliente, caso, "test-script", limite_s=30)
    assert r["estado"] == "DONE" and r["task_completed"] is True
    assert [a["decision"] for a in r["aprobaciones"]] == ["rechazado", "aprobado"]
    assert r["aprobaciones"][1]["servicios"] == ["localizacion_storyline"]
