"""Configuracion central del proyecto (evolucion de la de las pre-entregas 5 a 7).

Un unico lugar donde se leen las variables de entorno y se resuelven las rutas.
Ningun otro modulo llama a os.getenv(): si manana cambia el modelo, el
proveedor, el indice o un limite, se toca el .env y nada mas. Esto evita el
"hard-coding" que advierte la consigna (ids de indices o nombres de modelos
escritos en el codigo).

Novedades de la entrega final:
    - EMBEDDINGS (Gemini): convierten el corpus en vectores para el RAG.
      Son independientes del LLM que razona (Anthropic): Claude nunca ve un
      vector, solo el texto que la busqueda recupera.
    - CHROMA: la base vectorial corre como servidor (Docker) y se le habla
      con su cliente asincrono.
    - TAVILY: el buscador web del Agente Investigador.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raiz del repositorio (config.py esta en <raiz>/comun/config.py)
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

PROVEEDORES_VALIDOS: tuple[str, ...] = ("gemini", "anthropic")

VARIABLE_DE_CLAVE: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Version fijada a proposito: los alias tipo "latest" cambian solos y
# volverian irreproducible la entrega.
MODELO_POR_DEFECTO: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}

MODELO_EMBEDDINGS_POR_DEFECTO = "gemini-embedding-001"

MARCADORES_DE_EJEMPLO: tuple[str, ...] = ("pega_aca_tu_clave", "tu_clave", "sk-ant-xxx")


def _clave(variable: str) -> str:
    """Lee una clave del entorno sin comillas ni espacios."""
    return os.getenv(variable, "").strip().strip("\"'")


def _validar_clave(variable: str, valor: str, para_que: str) -> None:
    """Falla temprano y claro si una clave falta o es un valor de ejemplo."""
    if not valor:
        raise RuntimeError(
            f"Falta {variable} ({para_que}). Copia .env.example a .env y completa esa variable."
        )
    if valor in MARCADORES_DE_EJEMPLO or len(valor) < 20:
        raise RuntimeError(
            f"{variable} parece no ser una clave real. Abri el .env y "
            "reemplaza el valor por tu clave, sin comillas ni espacios."
        )


@dataclass(frozen=True)
class Configuracion:
    """Parametros de ejecucion. Inmutable: se arma una vez y nadie la toca."""

    # --- LLM que razona (Supervisor y agentes) -----------------------------
    proveedor: str
    api_key: str
    modelo: str
    temperatura: float
    max_rondas_supervisor: int
    # Pasos internos maximos de cada agente (su ciclo ReAct: cada llamada a una
    # herramienta son 2 pasos). La primera prueba real con 12 se quedo corta.
    limite_pasos_agente: int
    limite_recursion_grafo: int
    timeout_modelo: float
    ruta_logs: Path

    # --- RAG -----------------------------------------------------------------
    # Los embeddings son siempre de Gemini (Anthropic no ofrece embeddings).
    google_api_key: str = ""
    modelo_embeddings: str = MODELO_EMBEDDINGS_POR_DEFECTO
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    # Prefijo de las colecciones en Chroma: "headway" -> headway_catalogo, headway_metodologia.
    chroma_prefijo: str = "headway"
    ruta_corpus: Path = RAIZ_PROYECTO / "data" / "corpus"
    # Cuantos fragmentos devuelve cada busqueda.
    rag_top_k: int = 4

    # --- Busqueda web (Agente Investigador) ----------------------------------
    tavily_api_key: str = ""
    # Presupuesto de busquedas + lecturas web del investigador por intervencion.
    max_consultas_web: int = 8

    # --- API, worker y observabilidad (reutilizado de la PE7) ---------------
    redis_url: str = "redis://localhost:6379/0"
    max_trabajos_concurrentes: int = 5
    observabilidad: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    proyecto_phoenix: str = "headway-propuestas"

    @classmethod
    def desde_entorno(cls) -> "Configuracion":
        """Construye la configuracion leyendo el .env. Falla temprano y claro."""
        load_dotenv(RAIZ_PROYECTO / ".env")

        proveedor = os.getenv("PROVEEDOR", "anthropic").strip().lower()
        if proveedor not in PROVEEDORES_VALIDOS:
            raise RuntimeError(
                f"PROVEEDOR='{proveedor}' no es valido. Opciones: "
                f"{', '.join(PROVEEDORES_VALIDOS)}."
            )

        variable = VARIABLE_DE_CLAVE[proveedor]
        api_key = _clave(variable)
        _validar_clave(variable, api_key, f"LLM del proveedor '{proveedor}'")
        if proveedor == "anthropic" and not api_key.startswith("sk-ant-"):
            raise RuntimeError("ANTHROPIC_API_KEY deberia empezar con 'sk-ant-'.")

        # Los embeddings necesitan la clave de Google aunque el LLM sea Claude.
        google_api_key = _clave("GOOGLE_API_KEY")
        _validar_clave("GOOGLE_API_KEY", google_api_key, "embeddings del RAG")

        return cls(
            proveedor=proveedor,
            api_key=api_key,
            modelo=os.getenv("MODELO_LLM", "").strip() or MODELO_POR_DEFECTO[proveedor],
            temperatura=float(os.getenv("TEMPERATURA", "0")),
            max_rondas_supervisor=int(os.getenv("MAX_RONDAS_SUPERVISOR", "8")),
            limite_pasos_agente=int(os.getenv("LIMITE_PASOS_AGENTE", "25")),
            limite_recursion_grafo=int(os.getenv("LIMITE_RECURSION_GRAFO", "40")),
            timeout_modelo=float(os.getenv("TIMEOUT_MODELO", "90")),
            max_consultas_web=int(os.getenv("MAX_CONSULTAS_WEB", "8")),
            ruta_logs=RAIZ_PROYECTO / "logs",
            google_api_key=google_api_key,
            modelo_embeddings=os.getenv("MODELO_EMBEDDINGS", "").strip() or MODELO_EMBEDDINGS_POR_DEFECTO,
            chroma_host=os.getenv("CHROMA_HOST", "localhost").strip(),
            chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
            chroma_prefijo=os.getenv("CHROMA_PREFIJO_COLECCIONES", "headway").strip(),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            tavily_api_key=_clave("TAVILY_API_KEY"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(),
            max_trabajos_concurrentes=int(os.getenv("MAX_TRABAJOS_CONCURRENTES", "5")),
            observabilidad=os.getenv("OBSERVABILIDAD", "true").strip().lower() in ("1", "true", "si", "sí"),
            phoenix_endpoint=os.getenv(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
            ).strip(),
            proyecto_phoenix=os.getenv("PHOENIX_PROJECT_NAME", "headway-propuestas").strip(),
        )

    @classmethod
    def para_pruebas(cls, **cambios) -> "Configuracion":
        """Configuracion sin claves reales, para los tests con modelos simulados."""
        base = dict(
            proveedor="anthropic",
            api_key="clave-de-prueba-suficientemente-larga",
            modelo="modelo-simulado",
            temperatura=0.0,
            max_rondas_supervisor=8,
            limite_pasos_agente=25,
            limite_recursion_grafo=40,
            timeout_modelo=90.0,
            ruta_logs=RAIZ_PROYECTO / "logs" / "tests",
            google_api_key="clave-de-prueba-suficientemente-larga",
            modelo_embeddings="embeddings-simulados",
            chroma_prefijo="test",
        )
        base.update(cambios)
        return cls(**base)
