"""Las 5 pruebas oficiales del sistema, de punta a punta, contra la API.

Que hace:
    1. Lee los casos de scripts/pruebas.json (agencia + indicaciones + que
       responder en la aprobacion humana).
    2. Lanza los 5 pedidos a POST /propuestas, todos con la misma etiqueta
       (ej. "pruebas-20261012-153000"), y hace polling de cada job.
    3. Cuando un job pide aprobacion, responde segun el caso: "aprobar", o
       "rechazar_y_aprobar" (rechaza la primera version de los servicios con
       un comentario y aprueba la que el especialista arma despues). Asi una
       de las pruebas ejercita el ciclo completo del human-in-the-loop.
    4. Prueba ademas un pedido INVALIDO (agencia "<nombre>") y verifica que
       la API lo rechace con 422 sin ejecutar nada.
    5. Le pide a Phoenix las metricas de ESAS trazas (costo, tokens, latencia).
    6. Guarda todo en logs/pruebas_<etiqueta>.json y un resumen legible en
       logs/pruebas_<etiqueta>.md (sirve de evidencia para el README).

Requisitos: el sistema levantado con docker compose up.

Uso:
    python scripts/pruebas.py
    python scripts/pruebas.py --secuencial      # de a una (si el proveedor limita la concurrencia)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from scripts.metricas_phoenix import metricas_phoenix  # noqa: E402

ESTADOS_FINALES = {"DONE", "FAILED"}


async def correr_caso(cliente: httpx.AsyncClient, caso: dict, etiqueta: str, limite_s: float) -> dict:
    """Lanza un caso, responde las aprobaciones y espera el final."""
    t0 = time.perf_counter()
    cuerpo = {"agencia": caso["agencia"], "indicaciones": caso.get("indicaciones", ""), "etiqueta": etiqueta}
    if caso.get("sitio_web"):
        cuerpo["sitio_web"] = caso["sitio_web"]
    r = await cliente.post("/propuestas", json=cuerpo)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"[{caso['agencia']}] job {job_id[:8]} creado", flush=True)

    aprobaciones: list[dict] = []
    job: dict = {}
    while time.perf_counter() - t0 < limite_s:
        await asyncio.sleep(2)
        job = (await cliente.get(f"/propuestas/{job_id}")).json()
        if job["estado"] in ESTADOS_FINALES:
            break
        if job["estado"] == "AWAITING_APPROVAL":
            pedido = job["pedido_aprobacion"]
            primera = not aprobaciones
            rechazar = caso.get("aprobacion") == "rechazar_y_aprobar" and primera
            decision = (
                {"aprobado": False, "revisor": "script-pruebas", "comentario": caso.get("comentario", "")}
                if rechazar else {"aprobado": True, "revisor": "script-pruebas"}
            )
            aprobaciones.append({
                "version": pedido["version_match"],
                "servicios": [s["servicio_id"] for s in pedido["servicios"]],
                "decision": "rechazado" if rechazar else "aprobado",
            })
            print(f"[{caso['agencia']}] match v{pedido['version_match']} {[s['servicio_id'] for s in pedido['servicios']]}"
                  f" -> {'RECHAZADO' if rechazar else 'APROBADO'}", flush=True)
            (await cliente.post(f"/propuestas/{job_id}/aprobar", json=decision)).raise_for_status()
    else:
        job = {**job, "estado": "TIMEOUT", "error": f"No termino en {limite_s:.0f} s"}

    resultado = job.get("resultado") or {}
    propuesta = resultado.get("propuesta") or {}
    perfil = resultado.get("perfil") or {}
    print(f"[{caso['agencia']}] {job['estado']} en {time.perf_counter() - t0:.0f} s", flush=True)
    return {
        "agencia": caso["agencia"],
        "indicaciones": caso.get("indicaciones", ""),
        "job_id": job_id,
        "estado": job["estado"],
        "error": job.get("error"),
        "duracion_grafo_s": job.get("duracion_segundos"),
        "duracion_cliente_s": round(time.perf_counter() - t0, 1),
        "task_completed": resultado.get("task_completed"),
        "recorrido": resultado.get("recorrido"),
        "puntaje": perfil.get("puntaje"),
        "aprobaciones": aprobaciones,
        "canal": propuesta.get("canal"),
        "idioma": propuesta.get("idioma"),
        "asunto": propuesta.get("asunto"),
        "mensaje": propuesta.get("mensaje"),
    }


async def caso_invalido(cliente: httpx.AsyncClient) -> dict:
    """El pedido de la primera prueba real, ahora frenado por el contrato de entrada."""
    r = await cliente.post("/propuestas", json={"agencia": "<nombre>"})
    return {"pedido": {"agencia": "<nombre>"}, "status_http": r.status_code, "ok": r.status_code == 422}


def resumen_markdown(etiqueta: str, resultados: list[dict], invalido: dict, metricas: dict | None) -> str:
    lineas = [f"# Pruebas del sistema — `{etiqueta}`", "",
              "| # | Agencia | Estado | Completa | Recorrido | Puntaje | Aprobaciones | Canal | Duración (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(resultados, start=1):
        aprob = "; ".join(f"v{a['version']} {a['decision']}" for a in r["aprobaciones"]) or "-"
        lineas.append(f"| {n} | {r['agencia']} | {r['estado']} | {r['task_completed']} | "
                      f"{' → '.join(r['recorrido'] or [])} | {r['puntaje']} | {aprob} | {r['canal']} | "
                      f"{r['duracion_grafo_s']} |")
    lineas += ["", f"**Pedido inválido** (`agencia: \"<nombre>\"`): HTTP {invalido['status_http']} "
               f"{'✅ rechazado por Pydantic' if invalido['ok'] else '❌'}", ""]
    if metricas:
        lineas += ["## Métricas (Phoenix)", "",
                   f"- Trazas: {metricas['trazas']}",
                   f"- Costo total: US$ {metricas['costo_total_usd'] or 0:.4f} — por ejecución: "
                   f"US$ {metricas['costo_por_ejecucion_usd'] or 0:.4f}",
                   f"- Tokens: {metricas['tokens_entrada']:.0f} entrada / {metricas['tokens_salida']:.0f} salida",
                   f"- Latencia p50: {metricas['latencia_p50_s'] or 0:.1f} s — p95: {metricas['latencia_p95_s'] or 0:.1f} s",
                   f"- Filtro en Phoenix: `{metricas['filtro_en_phoenix']}`", ""]
    lineas += ["## Mensajes generados", ""]
    for r in resultados:
        lineas += [f"### {r['agencia']} ({r['canal']}, {r['idioma']})", ""]
        if r["asunto"]:
            lineas.append(f"**Asunto:** {r['asunto']}  ")
        lineas += ["```", r["mensaje"] or "(sin mensaje)", "```", ""]
    return "\n".join(lineas)


async def main() -> None:
    argumentos = argparse.ArgumentParser(description="Las 5 pruebas oficiales contra la API")
    argumentos.add_argument("--api", default="http://localhost:8000")
    argumentos.add_argument("--phoenix", default="http://localhost:6006")
    argumentos.add_argument("--proyecto", default="headway-propuestas")
    argumentos.add_argument("--limite", type=float, default=900, help="segundos maximos por caso")
    argumentos.add_argument("--secuencial", action="store_true")
    args = argumentos.parse_args()

    casos = json.loads((RAIZ / "scripts" / "pruebas.json").read_text(encoding="utf-8"))["casos"]
    etiqueta = f"pruebas-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"{len(casos)} casos | etiqueta: {etiqueta} | {'secuencial' if args.secuencial else 'concurrente'}\n")

    async with httpx.AsyncClient(base_url=args.api, timeout=30) as cliente:
        invalido = await caso_invalido(cliente)
        print(f"Pedido invalido -> HTTP {invalido['status_http']} ({'OK' if invalido['ok'] else 'ERROR'})\n")
        if args.secuencial:
            resultados = [await correr_caso(cliente, c, etiqueta, args.limite) for c in casos]
        else:
            resultados = await asyncio.gather(*[correr_caso(cliente, c, etiqueta, args.limite) for c in casos])

    print("\nConsultando metricas en Phoenix...")
    try:
        # Cada caso deja 1 traza, o 2+ si hubo aprobacion (inicio + cada reanudacion).
        esperadas = sum(1 + len(r["aprobaciones"]) for r in resultados)
        metricas = await metricas_phoenix(args.phoenix, args.proyecto, etiqueta, esperadas)
    except httpx.HTTPError as error:
        print(f"  No se pudo consultar Phoenix ({error}).")
        metricas = None

    carpeta = RAIZ / "logs"
    carpeta.mkdir(exist_ok=True)
    datos = {"etiqueta": etiqueta, "casos": resultados, "pedido_invalido": invalido, "phoenix": metricas}
    (carpeta / f"pruebas_{etiqueta}.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    md = carpeta / f"pruebas_{etiqueta}.md"
    md.write_text(resumen_markdown(etiqueta, resultados, invalido, metricas), encoding="utf-8")

    completas = sum(1 for r in resultados if r["task_completed"])
    print(f"\n{completas}/{len(resultados)} propuestas completas. Resumen: {md.relative_to(RAIZ)}")
    if metricas:
        print(f"Para las capturas, filtra en Phoenix con:  {metricas['filtro_en_phoenix']}")


if __name__ == "__main__":
    asyncio.run(main())
