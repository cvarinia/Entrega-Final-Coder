"""CLI: corre el sistema completo en la terminal (sin API ni Redis).

Uso:
    python main.py "Quiero ofrecerle nuestros servicios a <agencia>"
    python main.py --diagrama        # exporta docs/grafo.mmd (no usa la API)

Necesita Chroma levantado (docker compose up -d chroma) y las tres claves del
.env (Anthropic, Google y Tavily). Cuando el grafo se pausa para aprobar los
servicios, la pregunta aparece en la terminal. Es el mismo mecanismo que usa
la API (interrupt + Command(resume=...)), con un checkpointer en memoria.

Cada corrida deja una traza JSON en logs/.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command

from comun.config import RAIZ_PROYECTO, Configuracion
from comun.mensajes import texto_de_mensaje
from graph import checkpointer_en_memoria, config_de_ejecucion, construir_grafo, estado_inicial
from state import DecisionHumana



def exportar_diagrama() -> None:
    """Genera docs/grafo.mmd con modelo y recursos simulados (no necesita claves)."""
    from comun.modelo_simulado import ModeloSimulado
    from comun.recursos import Recursos

    recursos = Recursos(buscador=None, catalogo=None, web=None)  # type: ignore[arg-type]
    grafo = construir_grafo(Configuracion.para_pruebas(), recursos, modelo=ModeloSimulado())
    carpeta = RAIZ_PROYECTO / "docs"
    carpeta.mkdir(exist_ok=True)
    (carpeta / "grafo.mmd").write_text(grafo.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Diagrama guardado en {carpeta / 'grafo.mmd'}")


def _guardar_traza(config: Configuracion, pedido: str, final: dict[str, Any], segundos: float) -> Path:
    config.ruta_logs.mkdir(parents=True, exist_ok=True)
    ruta = config.ruta_logs / f"traza_{datetime.now():%Y%m%d_%H%M%S}.json"

    def volcar(valor: Any) -> Any:
        return valor.model_dump() if hasattr(valor, "model_dump") else valor

    traza = {
        "pedido": pedido,
        "modelo": f"{config.proveedor}/{config.modelo}",
        "embeddings": config.modelo_embeddings,
        "duracion_segundos": round(segundos, 1),
        "task_completed": final.get("task_completed"),
        "motivo_cierre": final.get("motivo_cierre"),
        "recorrido": [d.siguiente for d in final.get("decisiones", [])],
        "decisiones_supervisor": [volcar(d) for d in final.get("decisiones", [])],
        "perfil": volcar(final.get("perfil")),
        "match": volcar(final.get("match")),
        "decision_match": volcar(final.get("decision_match")),
        "propuesta": volcar(final.get("propuesta")),
        "aportes": [volcar(a) for a in final.get("aportes", [])],
        "respuesta_final": texto_de_mensaje(final["messages"][-1]),
    }
    ruta.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


async def _preguntar_aprobacion(pedido: dict[str, Any]) -> DecisionHumana:
    print("\n" + "-" * 70)
    print(f"APROBACION REQUERIDA (match v{pedido['version_match']}): {pedido['pregunta']}")
    lsp = pedido["lsp"]
    print(f"LSP: {lsp['nombre']} | puntaje {lsp['puntaje']}/11 | prioridad {lsp['prioridad_contacto']}")
    print(f"Argumento: {pedido['argumento_central']}")
    for s in pedido["servicios"]:
        adicional = " (adicional)" if s.get("tipo") == "add_on" else ""
        print(f"  - {s['servicio']}{adicional} [{s['prioridad']}, {s['tarifa']}]: {s['motivo']}")
    print("-" * 70)
    # input() bloquea: se corre en un hilo aparte para no frenar el event loop.
    respuesta = await asyncio.to_thread(input, "¿Aprobas estos servicios? (s/n): ")
    aprobado = respuesta.strip().lower() in ("s", "si", "sí", "y")
    comentario = "" if aprobado else await asyncio.to_thread(input, "¿Que cambiarias?: ")
    return DecisionHumana(aprobado=aprobado, revisor="cli", comentario=comentario.strip())


async def ejecutar(pedido: str) -> dict[str, Any]:
    from comun.recursos import abrir_recursos

    config = Configuracion.desde_entorno()
    print(f"Modelo: {config.proveedor}/{config.modelo} | Embeddings: {config.modelo_embeddings}")
    recursos = await abrir_recursos(config)
    grafo = construir_grafo(config, recursos, checkpointer=checkpointer_en_memoria())
    cfg = config_de_ejecucion(f"cli-{uuid.uuid4().hex[:8]}", config)
    print(f"Pedido: {pedido}\n")

    inicio = time.perf_counter()
    entrada: Any = estado_inicial(pedido)
    while True:
        async for actualizacion in grafo.astream(entrada, cfg, stream_mode="updates"):
            for nodo in actualizacion:
                if nodo != "__interrupt__":
                    print(f"<- termino: {nodo}\n", flush=True)
        snapshot = await grafo.aget_state(cfg)
        if not snapshot.interrupts:
            break
        decision = await _preguntar_aprobacion(snapshot.interrupts[0].value)
        entrada = Command(resume=decision.model_dump())  # mismo thread_id
    final = (await grafo.aget_state(cfg)).values
    segundos = time.perf_counter() - inicio

    print("=" * 70)
    print(texto_de_mensaje(final["messages"][-1]))
    print("=" * 70)
    print(f"Recorrido: {' -> '.join(d.siguiente for d in final['decisiones'])}")
    print(f"Tarea completa: {final['task_completed']} | {final['motivo_cierre']}")
    print(f"Duracion: {segundos:.0f} s | Traza: {_guardar_traza(config, pedido, final, segundos)}")
    return final


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    if argumentos == ["--diagrama"]:
        exportar_diagrama()
    elif not argumentos:
        print('Uso: python main.py "Quiero ofrecerle nuestros servicios a la agencia <nombre>"')
    elif "<" in " ".join(argumentos) or ">" in " ".join(argumentos):
        print("El pedido tiene un marcador como <nombre>: reemplazalo por el nombre real de la agencia.")
    else:
        asyncio.run(ejecutar(" ".join(argumentos)))
