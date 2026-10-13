"""API REST del sistema de propuestas (evolucion de la de la PE7).

La API NUNCA ejecuta el grafo: valida el pedido, lo registra, lo encola en
Redis y devuelve un job_id en milisegundos. El trabajo lo hace el worker
(python -m app.worker), que es otro proceso.

Endpoints:
    POST /propuestas                   -> crea el job (202 + job_id)
    GET  /propuestas/{job_id}          -> estado del job (polling)
    POST /propuestas/{job_id}/aprobar  -> respuesta humana sobre los servicios propuestos
    GET  /health                       -> ¿la API y Redis responden?

Validacion de ENTRADA con Pydantic (leccion de la primera prueba real): el
pedido ya no es texto libre. `agencia` es obligatoria y no acepta marcadores
como "<nombre>". Si falta, la API responde 422 antes de gastar una sola
llamada a un modelo.

Documentacion interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.jobs import Job, RepositorioJobs
from comun.config import Configuracion
from state import DecisionHumana


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    """Abre la conexion a Redis al arrancar y la cierra al apagar."""
    config = Configuracion.desde_entorno()
    redis = Redis.from_url(config.redis_url, decode_responses=True)
    await redis.ping()  # si Redis no esta levantado, la API no arranca (falla clara)
    app.state.redis = redis
    app.state.jobs = RepositorioJobs(redis)
    yield
    await redis.aclose()


app = FastAPI(
    title="Headway - Propuestas para agencias de localizacion",
    description=(
        "Sistema multi-agente (Supervisor + investigador + especialista + redactor) que investiga "
        "una agencia de localizacion, elige servicios de Headway con aprobacion humana y redacta "
        "el primer contacto."
    ),
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


# --------------------------------------------------------------------------
# Contratos de entrada y salida
# --------------------------------------------------------------------------


class PedidoPropuesta(BaseModel):
    agencia: str = Field(
        min_length=2, max_length=120, pattern=r"^[^<>]+$",
        description="Nombre de la agencia de localizacion (LSP). Obligatorio.",
        examples=["Vistatec"],
    )
    sitio_web: str | None = Field(
        default=None, pattern=r"^https?://\S+$", max_length=300,
        description="Opcional: ayuda al investigador si hay agencias con nombres parecidos.",
    )
    indicaciones: str = Field(
        default="", max_length=600,
        description="Opcional: que te interesa ofrecer, idioma del mensaje, etc.",
        examples=["Me interesa ofrecer localizacion de Storyline."],
    )
    etiqueta: str | None = Field(
        default=None, max_length=60, pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Opcional. Agrupa jobs en el dashboard de trazas (por ejemplo, 'pruebas-1').",
    )

    def como_pedido(self) -> str:
        """El texto que recibe el Supervisor como pedido de la persona."""
        sitio = f" ({self.sitio_web})" if self.sitio_web else ""
        return f"Quiero ofrecerle los servicios de Headway a la agencia {self.agencia}{sitio}. {self.indicaciones}".strip()


class TareaCreada(BaseModel):
    job_id: str
    estado: str
    consultar_en: str


def _jobs(request: Request) -> RepositorioJobs:
    return request.app.state.jobs


async def _job_o_404(request: Request, job_id: str) -> Job:
    job = await _jobs(request).obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    return job


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.post("/propuestas", status_code=status.HTTP_202_ACCEPTED, response_model=TareaCreada)
async def crear_propuesta(cuerpo: PedidoPropuesta, request: Request) -> TareaCreada:
    """Encola el pedido y devuelve el job_id al instante (202 = aceptado, en proceso)."""
    job = await _jobs(request).crear(cuerpo.como_pedido(), cuerpo.etiqueta, agencia=cuerpo.agencia)
    return TareaCreada(job_id=job.job_id, estado=job.estado, consultar_en=f"/propuestas/{job.job_id}")


@app.get("/propuestas/{job_id}", response_model=Job)
async def consultar_propuesta(job_id: str, request: Request) -> Job:
    """Estado del job. En AWAITING_APPROVAL incluye los servicios a aprobar."""
    return await _job_o_404(request, job_id)


@app.post("/propuestas/{job_id}/aprobar", status_code=status.HTTP_202_ACCEPTED, response_model=Job)
async def aprobar_servicios(job_id: str, decision: DecisionHumana, request: Request) -> Job:
    """Aprobar o rechazar (con comentario) los servicios propuestos.

    El cuerpo se valida con el mismo contrato (DecisionHumana) que usa el nodo
    de aprobacion del grafo: lo que no cumple, se rechaza aca con un 422.
    """
    job = await _job_o_404(request, job_id)
    if job.estado != "AWAITING_APPROVAL":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job esta en {job.estado}: solo se puede aprobar en AWAITING_APPROVAL.",
        )
    version = (job.pedido_aprobacion or {}).get("version_match", 0)
    if not await _jobs(request).responder_aprobacion(job_id, decision, version):
        raise HTTPException(status.HTTP_409_CONFLICT, "Esta version de los servicios ya fue respondida.")
    return await _job_o_404(request, job_id)


@app.get("/health")
async def salud(request: Request) -> dict[str, str]:
    await request.app.state.redis.ping()
    return {"api": "ok", "redis": "ok"}
