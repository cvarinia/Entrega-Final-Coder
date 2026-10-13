"""Estado compartido del grafo y contratos de datos (Pydantic).

Tres ideas que vienen de las pre-entregas 6 y 7 y se mantienen:

1. CONTRATOS VALIDADOS. Lo que entrega cada agente no es texto libre: es un
   modelo Pydantic (PerfilLSP, MatchServicios, Propuesta). El agente lo
   entrega llamando a una herramienta `entregar_...` cuyos argumentos se
   validan contra el modelo. Si algo no cumple, la herramienta devuelve el
   error y el agente corrige en el momento, no tres nodos despues.

2. AISLAMIENTO DE ESCRITURA. Cada nodo escribe solo su parcela del estado:
   el Supervisor sus decisiones, el investigador el perfil, el especialista
   el match, el redactor la propuesta, la aprobacion la decision humana.

3. PROVENIENCIA. `aportes` es una lista acumulativa (reducer operator.add):
   cada intervencion queda registrada con la evidencia real de sus
   herramientas (que busco, que URL leyo, que devolvio). Asi se puede
   verificar, por ejemplo, que una fuente citada fue realmente consultada.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, computed_field

# --------------------------------------------------------------------------
# Nombres de agentes y destinos del Supervisor
# --------------------------------------------------------------------------

Especialista = Literal["investigador", "especialista", "redactor"]
Destino = Literal["investigador", "especialista", "redactor", "FINALIZAR"]

URL = Field(pattern=r"^https?://\S+$", max_length=500)


# --------------------------------------------------------------------------
# Perfil de la LSP (lo entrega el Investigador)
# --------------------------------------------------------------------------

CriterioLSP = Literal[
    "servicio_elearning",
    "trabaja_con_articulate",
    "recluta_proveedores",
    "muchos_idiomas",
    "en_crecimiento",
    "ofrece_dtp_multimedia",
]

# Puntos por criterio. Es la misma tabla del documento de metodologia
# 04_criterios_lsp_ideal.md: el LLM la lee para saber QUE buscar, pero el
# puntaje lo calcula este codigo, no el modelo (un LLM no es confiable sumando).
PUNTOS_CRITERIO: dict[str, int] = {
    "servicio_elearning": 3,
    "trabaja_con_articulate": 3,
    "recluta_proveedores": 2,
    "muchos_idiomas": 1,
    "en_crecimiento": 1,
    "ofrece_dtp_multimedia": 1,
}


class Senal(BaseModel):
    """Un indicio publico de que la LSP podria necesitar a Headway."""

    criterio: CriterioLSP
    descripcion: str = Field(min_length=10, max_length=400, description="Que se encontro, en una o dos frases.")
    fuente_url: str = URL


TipoCanal = Literal["portal_proveedores", "email", "linkedin", "formulario_web"]


class CanalContacto(BaseModel):
    """Un canal publico para contactar a la LSP (nunca datos personales)."""

    tipo: TipoCanal
    descripcion: str = Field(
        min_length=5, max_length=300,
        description="Por ejemplo 'Formulario Join our vendor network' o 'vendors@empresa.com (direccion publica)'.",
    )
    url: str | None = Field(default=None, pattern=r"^https?://\S+$", max_length=500)


class PerfilLSP(BaseModel):
    """Lo que el investigador averiguo de la agencia."""

    nombre: str = Field(min_length=2, max_length=120)
    sitio_web: str = URL
    resumen: str = Field(min_length=20, max_length=800)
    verticales: list[str] = Field(default_factory=list, max_length=20)
    senales: list[Senal] = Field(min_length=1, max_length=12)
    canales_contacto: list[CanalContacto] = Field(default_factory=list, max_length=6)

    @computed_field  # se calcula con codigo y viaja junto al perfil
    @property
    def puntaje(self) -> int:
        """Suma de puntos, contando cada criterio una sola vez (maximo 11)."""
        return sum(PUNTOS_CRITERIO[c] for c in {s.criterio for s in self.senales})

    @computed_field
    @property
    def prioridad_contacto(self) -> Literal["alta", "media", "baja"]:
        return "alta" if self.puntaje >= 8 else "media" if self.puntaje >= 5 else "baja"

    def urls_citadas(self) -> set[str]:
        urls = {self.sitio_web, *(s.fuente_url for s in self.senales)}
        urls |= {c.url for c in self.canales_contacto if c.url}
        return urls


# --------------------------------------------------------------------------
# Match de servicios (lo entrega el Especialista; lo aprueba una persona)
# --------------------------------------------------------------------------


class ServicioElegido(BaseModel):
    """Lo que elige el LLM: que servicio y por que."""

    servicio_id: str = Field(min_length=3, max_length=60)
    motivo: str = Field(min_length=10, max_length=400, description="Que senal de la LSP responde este servicio.")


class ServicioPropuesto(ServicioElegido):
    """Lo anterior + los datos del catalogo, completados por CODIGO (no por el LLM)."""

    servicio: str
    prioridad: str
    tarifa: str
    tipo: str = Field(default="servicio", description="'servicio' (principal) o 'add_on' (adicional).")


class MatchServicios(BaseModel):
    servicios: list[ServicioPropuesto] = Field(min_length=1, max_length=4)
    argumento_central: str = Field(min_length=20, max_length=600)


# --------------------------------------------------------------------------
# Propuesta (la entrega el Redactor)
# --------------------------------------------------------------------------

CanalPropuesta = Literal["portal_proveedores", "email", "linkedin"]
Pieza = Literal["brochure", "carta_presentacion", "hoja_tarifas", "oferta_prueba"]


class Propuesta(BaseModel):
    canal: CanalPropuesta
    motivo_canal: str = Field(min_length=10, max_length=400)
    idioma: str = Field(min_length=2, max_length=30, description="Idioma del mensaje, por ejemplo 'ingles'.")
    asunto: str = Field(default="", max_length=120, description="Obligatorio salvo en LinkedIn.")
    mensaje: str = Field(min_length=40, max_length=2000, description="El primer mensaje, listo para enviar.")
    piezas: list[Pieza] = Field(min_length=1, max_length=4)
    seguimiento: str = Field(min_length=10, max_length=600, description="Que hacer si la LSP responde.")


# --------------------------------------------------------------------------
# Supervisor y decision humana
# --------------------------------------------------------------------------


class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor."""

    evaluacion: str = Field(description="Una o dos frases: que se cumple y que falta segun la rubrica.")
    siguiente: Destino = Field(description="Quien interviene ahora, o FINALIZAR.")
    instruccion: str = Field(
        default="",
        description=(
            "Instruccion concreta y autocontenida para el agente elegido. Si es una "
            "correccion, decir exactamente que corregir. Vacia si FINALIZAR."
        ),
    )


class RegistroDecision(DecisionSupervisor):
    ronda: int
    forzada: bool = Field(default=False, description="True si la tomo el codigo (limite de rondas).")


class DecisionHumana(BaseModel):
    """Respuesta de la persona que revisa el match de servicios.

    Se valida dos veces: en la API (al recibir el POST) y en el nodo de
    aprobacion (al reanudar el grafo).
    """

    aprobado: bool
    revisor: str = Field(default="anonimo", min_length=1, max_length=80)
    comentario: str = Field(default="", max_length=500, description="Si se rechaza: que cambiar.")


# --------------------------------------------------------------------------
# Proveniencia
# --------------------------------------------------------------------------


class Evidencia(BaseModel):
    """Una llamada a herramienta, con su resultado real."""

    herramienta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)
    resultado: Any = None
    error: str | None = None


class Aporte(BaseModel):
    """Una intervencion de un agente."""

    agente: Especialista
    ronda: int = Field(ge=1)
    instruccion: str
    resumen: str
    entrego: bool = Field(description="True si el agente completo su entrega formal (entregar_...).")
    evidencia: list[Evidencia] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Estado del grafo
# --------------------------------------------------------------------------


class EstadoPropuesta(MessagesState):
    """Estado global. `messages` (heredado) guarda solo la conversacion visible."""

    # --- Supervisor ---------------------------------------------------------
    next_agent: Destino
    instruccion_actual: str
    ronda: int
    decisiones: Annotated[list[RegistroDecision], operator.add]
    task_completed: bool
    motivo_cierre: str

    # --- Agentes (cada uno su parcela; la ultima entrega valida pisa a la anterior)
    aportes: Annotated[list[Aporte], operator.add]
    perfil: PerfilLSP | None
    ronda_perfil: int
    match: MatchServicios | None
    ronda_match: int
    version_match: int            # sube cada vez que el especialista entrega un match nuevo
    propuesta: Propuesta | None
    version_match_propuesta: int  # sobre que version del match se escribio la propuesta

    # --- Human-in-the-loop -------------------------------------------------
    decision_match: DecisionHumana | None
    version_aprobada: int         # 0 = ningun match aprobado todavia


# Tipos propios que el checkpointer tiene que poder reconstruir. LangGraph
# solo deserializa clases declaradas explicitamente (medida de seguridad).
TIPOS_DEL_ESTADO: list[tuple[str, str]] = [
    (clase.__module__, clase.__name__)
    for clase in (
        Evidencia, Aporte, RegistroDecision, DecisionHumana,
        Senal, CanalContacto, PerfilLSP,
        ServicioPropuesto, MatchServicios, Propuesta,
    )
]
